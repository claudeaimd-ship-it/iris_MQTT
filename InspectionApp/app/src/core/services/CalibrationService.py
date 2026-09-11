import json
import os
from datetime import date, datetime
from typing import Callable

import numpy as np

from app.src.core.models.CalibrationModels import (
    BlockSweepResult,
    CalibrationResult,
    SweepResult,
)
from app.src.interfaces.IPaDiMFeatureExtractor import IPaDiMFeatureExtractor


class CalibrationService:
    """
    Core service for PaDiM model calibration.

    Orchestrates feature extraction, Gaussian fitting, score computation, and
    output file writing for one or more views. This service has no dependency
    on any adapter — all feature extraction is delegated to the injected
    ``IPaDiMFeatureExtractor``.

    Two entry points are provided:

    * ``run_sweep()`` — evaluates every configured backbone block (b3–b17) for
      each view and returns ``SweepResult`` objects. Does NOT write any files.
    * ``run_calibration()`` — fits the final Gaussian for the chosen block,
      writes all output files to ``model_path``, and returns a
      ``CalibrationResult`` per view.

    A ``progress_cb`` callback is accepted by both methods so that
    ``GuiCalibrationAdapter`` can push live progress to ``IrisState`` without
    coupling the service to Flask.

    Attributes:
        _extractor (IPaDiMFeatureExtractor): Injected feature extractor.

    Note:
        Requires ``scikit-learn`` and ``scipy`` on the execution host. These
        are NOT listed in the base ``requirements.txt`` because they are only
        needed for calibration, not inference.
    """

    # Default block range for the sweep (inclusive on both ends).
    _DEFAULT_SWEEP_BLOCKS: tuple[int, int] = (3, 17)

    # Minimum sep_ratio a block must reach to be considered a valid candidate
    # for the "most consistent OK distribution" (lowest cv_ok) selection rule.
    # Blocks below this gate are excluded even if their cv_ok is the lowest,
    # since a low sep_ratio means NOK images are not reliably separated.
    _DEFAULT_MIN_SEP_GATE: float = 1.0

    def __init__(self, extractor: IPaDiMFeatureExtractor) -> None:
        """
        Args:
            extractor (IPaDiMFeatureExtractor): Feature extractor adapter.
        """
        self._extractor = extractor

    # =========================================================================
    # Public API
    # =========================================================================

    def run_sweep(
        self,
        view_configs: list[dict],
        image_dirs: dict[str, dict[str, str]],
        backbones_dir: str,
        blocks: tuple[int, int] | None = None,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[SweepResult] | None:
        """
        Evaluate every backbone block for all views and return sweep results.

        Does NOT write any files to disk. The caller (``GuiCalibrationAdapter``)
        is responsible for persisting results to ``IrisState``.

        Args:
            view_configs (list[dict]): List of view dicts, each with at least:
                ``view_name`` (str), ``roi`` (dict with x/y/w/h),
                ``training_shape`` (tuple[int, int]).
            image_dirs (dict[str, dict[str, str]]): Mapping from view_name to
                a dict with keys ``train_ok``, ``test_ok``, ``test_nok`` —
                each pointing to an image directory.
            backbones_dir (str): Directory containing the ONNX backbone files
                named ``teacher_mobilenetv2_backbone_b{N}.onnx``.
            blocks (tuple[int, int] | None): ``(first_block, last_block)``
                inclusive. Defaults to ``(3, 17)``.
            params (dict | None): Optional scoring parameters. Supported keys:
                ``top_k_pixels`` (int, default 2),
                ``border_crop_px`` (int, default 1),
                ``gaussian_sigma`` (float, default 0.0),
                ``use_clahe`` (bool, default False),
                ``padim_dim`` (int, default 100),
                ``padim_lambda`` (float, default 0.01),
                ``min_sep_gate`` (float, default 1.0) — minimum ``sep_ratio``
                a block must reach to be eligible for the "lowest cv_ok"
                selection rule in ``_build_sweep_result``. Blocks below this
                gate are only used as a fallback (highest ``sep_ratio``) if no
                block passes it.
            progress_cb (Callable[[int, int, str], None] | None): Called as
                ``progress_cb(current_step, total_steps, message)`` at each
                significant milestone.

        Returns:
            list[SweepResult]: One ``SweepResult`` per view, in the same order
                as ``view_configs``.
        """
        params        = params or {}
        first, last   = blocks or self._DEFAULT_SWEEP_BLOCKS
        block_range   = list(range(first, last + 1))
        total_steps   = len(view_configs) * len(block_range)
        step          = 0
        min_sep_gate  = params.get("min_sep_gate", self._DEFAULT_MIN_SEP_GATE)
        sweep_results: list[SweepResult] = []

        for view_cfg in view_configs:
            view_name     = view_cfg["view_name"]
            roi           = view_cfg["roi"]
            training_shape = view_cfg["training_shape"]
            dirs          = image_dirs[view_name]

            block_results: list[BlockSweepResult] = []

            for block in block_range:
                step += 1
                backbone_path = os.path.join(
                    backbones_dir,
                    f"teacher_mobilenetv2_backbone_b{block}.onnx",
                )
                if not os.path.isfile(backbone_path):
                    if progress_cb:
                        progress_cb(step, total_steps,
                                    f"{view_name} b{block}: backbone not found — skipped.")
                    continue

                if progress_cb:
                    progress_cb(step, total_steps,
                                f"{view_name} b{block}: extracting features…")

                self._extractor.load_backbone(backbone_path)

                train_ok_features = self._extract_all(
                    dirs["train_ok"], roi, training_shape, params,
                    view_cfg.get("masks", []), cancel_check
                )
                if train_ok_features is None:
                    if progress_cb:
                        progress_cb(step, total_steps, "Sweep cancelled by user.")
                    return None
                if len(train_ok_features) == 0:
                    if progress_cb:
                        progress_cb(step, total_steps,
                                    f"{view_name} b{block}: no train OK images — skipped.")
                    continue

                mean, precision, random_idx = self._fit_gaussian(
                    train_ok_features, params
                )

                test_ok_features  = self._extract_all(
                    dirs["test_ok"], roi, training_shape, params,
                    view_cfg.get("masks", []), cancel_check
                )
                if test_ok_features is None:
                    if progress_cb:
                        progress_cb(step, total_steps, "Sweep cancelled by user.")
                    return None

                test_nok_features = self._extract_all(
                    dirs["test_nok"], roi, training_shape, params,
                    view_cfg.get("masks", []), cancel_check
                )
                if test_nok_features is None:
                    if progress_cb:
                        progress_cb(step, total_steps, "Sweep cancelled by user.")
                    return None

                ok_scores  = self._score_all(test_ok_features,  mean, precision,
                                             random_idx, params)
                nok_scores = self._score_all(test_nok_features, mean, precision,
                                             random_idx, params)

                if len(ok_scores) == 0 or len(nok_scores) == 0:
                    if progress_cb:
                        progress_cb(step, total_steps,
                                    f"{view_name} b{block}: insufficient test images — skipped.")
                    continue

                auc       = self._compute_auc(ok_scores, nok_scores)
                max_ok    = float(np.max(ok_scores))
                min_nok   = float(np.min(nok_scores))
                sep_ratio = min_nok / max_ok if max_ok > 0 else 0.0

                min_ok  = float(np.min(ok_scores))
                mean_ok = float(np.mean(ok_scores))
                std_ok  = float(np.std(ok_scores))
                cv_ok   = std_ok / mean_ok if mean_ok > 0 else 0.0

                block_results.append(BlockSweepResult(
                    block=block,
                    auc=auc,
                    sep_ratio=sep_ratio,
                    min_nok=min_nok,
                    max_ok=max_ok,
                    min_ok=min_ok,
                    mean_ok=mean_ok,
                    std_ok=std_ok,
                    cv_ok=cv_ok,
                    n_train_ok=len(train_ok_features),
                    n_test_ok=len(ok_scores),
                    n_test_nok=len(nok_scores),
                ))

                if progress_cb:
                    progress_cb(step, total_steps,
                                f"{view_name} b{block}: AUC={auc:.4f} Sep={sep_ratio:.3f}x")

                if cancel_check and cancel_check():
                    if progress_cb:
                        progress_cb(step, total_steps, "Sweep cancelled by user.")
                    return None

            result = self._build_sweep_result(view_name, block_results, min_sep_gate)
            sweep_results.append(result)

        return sweep_results

    def run_calibration(
        self,
        view_configs: list[dict],
        image_dirs: dict[str, dict[str, str]],
        backbones_dir: str,
        blocks: dict[str, int],
        model_path: str,
        params: dict | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
    ) -> list[CalibrationResult]:
        """
        Fit the final PaDiM model for all views and write all output files.

        Each view uses the MobileNetV2 block specified in ``blocks``. This
        allows each view to use the block that produced the best separation
        ratio in the sweep, rather than a single shared block.

        Output files written per view:
        * ``padim_{view_name}_params.npz`` — Gaussian parameters.
        * ``thresholds.json`` — per-view min/max acceptance thresholds
          (merged with any existing entries).
        * ``eval_config.json`` — scoring params and calibration metadata.

        Args:
            view_configs (list[dict]): Same format as ``run_sweep()``.
            image_dirs (dict[str, dict[str, str]]): Same format as ``run_sweep()``.
            backbones_dir (str): Directory with ONNX backbone files.
            blocks (dict[str, int]): Mapping from view_name to the chosen
                MobileNetV2 block (e.g. ``{"front_view_A": 9, "side_view_B": 7}``).
                Populated automatically from sweep results by the caller.
            model_path (str): Directory where output files are written.
            params (dict | None): Same optional scoring params as ``run_sweep()``.
            progress_cb (Callable[[int, int, str], None] | None): Progress callback.

        Returns:
            list[CalibrationResult]: One result per view.
        """
        params      = params or {}
        total_steps = len(view_configs) * 3   # extract → fit → score per view
        step        = 0
        os.makedirs(model_path, exist_ok=True)
        graphs_dir = os.path.join(model_path, "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

        calibration_results: list[CalibrationResult] = []
        all_thresholds: dict[str, dict[str, float]] = {}

        # Load existing thresholds so we only overwrite entries for the views
        # being calibrated and keep any others intact.
        thresholds_path = os.path.join(model_path, "thresholds.json")
        if os.path.isfile(thresholds_path):
            with open(thresholds_path, "r", encoding="utf-8") as f:
                all_thresholds = json.load(f)

        for view_cfg in view_configs:
            view_name      = view_cfg["view_name"]
            roi            = view_cfg["roi"]
            training_shape = view_cfg["training_shape"]
            dirs           = image_dirs[view_name]

            # Each view may use a different backbone block.
            block         = blocks.get(view_name, 9)
            backbone_path = os.path.join(
                backbones_dir,
                f"teacher_mobilenetv2_backbone_b{block}.onnx",
            )

            step += 1
            if progress_cb:
                progress_cb(step, total_steps,
                            f"{view_name} (b{block}): extracting train features…")

            self._extractor.load_backbone(backbone_path)

            train_ok_features = self._extract_all(
                dirs["train_ok"], roi, training_shape, params,
                view_cfg.get("masks", [])
            )

            step += 1
            if progress_cb:
                progress_cb(step, total_steps,
                            f"{view_name}: fitting Gaussian (n={len(train_ok_features)})…")

            mean, precision, random_idx = self._fit_gaussian(train_ok_features, params)

            step += 1
            if progress_cb:
                progress_cb(step, total_steps,
                            f"{view_name}: scoring test set…")

            test_ok_features, test_ok_images = self._extract_all_with_images(
                dirs["test_ok"], roi, training_shape, params,
                view_cfg.get("masks", [])
            )
            test_nok_features, test_nok_images = self._extract_all_with_images(
                dirs["test_nok"], roi, training_shape, params,
                view_cfg.get("masks", [])
            )

            ok_scores,  ok_error_maps  = self._score_with_error_maps(
                test_ok_features,  mean, precision, random_idx, params
            )
            nok_scores, nok_error_maps = self._score_with_error_maps(
                test_nok_features, mean, precision, random_idx, params
            )

            auc       = self._compute_auc(ok_scores, nok_scores) if len(nok_scores) > 0 else 0.0
            max_ok    = float(np.max(ok_scores))  if len(ok_scores) > 0  else 0.0
            min_nok   = float(np.min(nok_scores)) if len(nok_scores) > 0 else 0.0
            sep_ratio = min_nok / max_ok if max_ok > 0 else 0.0

            threshold_max = float(np.percentile(ok_scores, 99.5)) * 1.10 if len(ok_scores) > 0 else 0.0
            threshold_min = float(np.min(ok_scores)) * 0.90              if len(ok_scores) > 0 else 0.0

            # Write per-view score distribution graph and heatmaps.
            self._save_calibration_graphs(
                view_name=view_name,
                ok_scores=ok_scores,
                nok_scores=nok_scores,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                ok_images=test_ok_images,
                nok_images=test_nok_images,
                ok_error_maps=ok_error_maps,
                nok_error_maps=nok_error_maps,
                graphs_dir=graphs_dir,
                timestamp=timestamp,
            )

            # Write per-view params file.
            params_path = os.path.join(model_path, f"padim_{view_name}_params.npz")
            np.savez_compressed(
                params_path,
                mean=mean,
                precision=precision,
                random_idx=random_idx,
            )

            all_thresholds[view_name] = {
                "min": threshold_min,
                "max": threshold_max,
            }

            calibration_results.append(CalibrationResult(
                view_name=view_name,
                block=block,  # per-view block used for this fit
                sep_ratio=sep_ratio,
                auc=auc,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                params_path=os.path.abspath(params_path),
                eval_config_path=os.path.abspath(
                    os.path.join(model_path, "eval_config.json")
                ),
                thresholds_path=os.path.abspath(thresholds_path),
                inference_type=view_cfg.get("inference_type", "standard"),
            ))

            if progress_cb:
                progress_cb(step, total_steps,
                            f"{view_name}: done — AUC={auc:.4f} Sep={sep_ratio:.3f}x "
                            f"thr=[{threshold_min:.4f}, {threshold_max:.4f}]")

        # Write thresholds.json (merged).
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(all_thresholds, f, indent=4)

        # Write eval_config.json — per-view blocks and best sep_ratio stored.
        # Merge with any existing config so that a partial recalibration
        # (e.g. only one view) does not discard the params of other views.
        best_sep = max((r.sep_ratio for r in calibration_results), default=0.0)
        abs_model_path = os.path.abspath(model_path)
        new_backbone_paths = {
            r.view_name: os.path.relpath(
                os.path.abspath(
                    os.path.join(backbones_dir, f"teacher_mobilenetv2_backbone_b{r.block}.onnx")
                ),
                abs_model_path,
            )
            for r in calibration_results
        }
        new_blocks = {r.view_name: r.block for r in calibration_results}
        new_inference_types = {r.view_name: r.inference_type for r in calibration_results}

        # Load existing config and merge so untouched views are preserved.
        eval_config_path = os.path.join(model_path, "eval_config.json")
        existing_config: dict = {}
        if os.path.isfile(eval_config_path):
            try:
                with open(eval_config_path, "r", encoding="utf-8") as _f:
                    existing_config = json.load(_f)
            except Exception:
                pass

        merged_blocks         = {**existing_config.get("blocks", {}),         **new_blocks}
        merged_backbone_paths = {**existing_config.get("backbone_paths", {}), **new_backbone_paths}
        merged_inference_types = {**existing_config.get("inference_types", {}), **new_inference_types}

        eval_config = {
            "top_k_pixels":     params.get("top_k_pixels",   existing_config.get("top_k_pixels",   2)),
            "border_crop_px":   params.get("border_crop_px", existing_config.get("border_crop_px", 1)),
            "gaussian_sigma":   params.get("gaussian_sigma", existing_config.get("gaussian_sigma", 0.0)),
            "use_clahe":        params.get("use_clahe",      existing_config.get("use_clahe",      False)),
            "blocks":           merged_blocks,
            "backbone_paths":   merged_backbone_paths,
            "inference_types":  merged_inference_types,
            "sep_ratio":        round(best_sep, 4),
            "calibration_date": date.today().isoformat(),
        }
        with open(eval_config_path, "w", encoding="utf-8") as f:
            json.dump(eval_config, f, indent=4)

        print(
            f"[OK] CalibrationService: calibration complete — "
            f"{len(calibration_results)} view(s) written to '{model_path}'."
        )
        return calibration_results

    # =========================================================================
    # Private — feature extraction helpers
    # =========================================================================

    def _extract_all(
        self,
        image_dir: str,
        roi: dict,
        training_shape: tuple[int, int],
        params: dict,
        masks: list[dict] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> list[np.ndarray] | None:
        """
        Load all JPEG/PNG images from a directory, apply masks + ROI crop + resize,
        optionally apply CLAHE, and extract features.

        Masks (put_black_circle, put_black_rectangle) are applied to the
        full-resolution image BEFORE the ROI crop, matching the canonical order
        enforced by SequenceSettings at inference time.

        Args:
            image_dir (str): Directory of images (e.g. ``train/OK``).
            roi (dict): ``{x, y, w, h}`` crop in original capture coordinates.
            training_shape (tuple[int, int]): ``(height, width)`` to resize to.
            params (dict): Scoring params — reads ``use_clahe``.
            masks (list[dict] | None): Optional list of mask tool dicts from the
                pipeline (``put_black_circle``, ``put_black_rectangle``). Applied
                before the ROI crop in canonical order.
            cancel_check (Callable[[], bool] | None): Called before processing
                each image. Returns ``None`` immediately if it returns ``True``.

        Returns:
            list[np.ndarray] | None: Feature maps, one per image, shape
                ``(H', W', C')``. Returns ``None`` if cancelled.
        """
        import cv2

        use_clahe = params.get("use_clahe", False)
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        target_h, target_w = training_shape
        features: list[np.ndarray] = []

        if not os.path.isdir(image_dir):
            return features

        for filename in sorted(os.listdir(image_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            if cancel_check and cancel_check():
                return None
            img_bgr = cv2.imread(os.path.join(image_dir, filename))
            if img_bgr is None:
                continue

            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb  = self._apply_masks(img_rgb, masks or [])
            cropped  = img_rgb[y:y + h, x:x + w]
            resized  = cv2.resize(cropped, (target_w, target_h))

            if use_clahe:
                resized = self._apply_clahe(resized)

            feature_map = self._extractor.extract_features(resized)
            features.append(feature_map)

        return features

    @staticmethod
    def _apply_masks(image_rgb: np.ndarray, masks: list[dict]) -> np.ndarray:
        """
        Apply put_black_circle and put_black_rectangle mask tools to a full-resolution
        image, matching the canonical pipeline order used by SequenceSettings.

        Args:
            image_rgb (np.ndarray): RGB image ``(H, W, 3)`` uint8.
            masks (list[dict]): Tool dicts from the sequence pipeline, each with
                ``"tool"`` (str) and ``"parameters"`` (dict).

        Returns:
            np.ndarray: Image with masks applied in-place (same shape and dtype).
        """
        import cv2

        if not masks:
            return image_rgb

        result = image_rgb.copy()
        for mask_def in masks:
            tool = mask_def.get("tool", "")
            p    = mask_def.get("parameters", {})
            if tool == "put_black_circle":
                cv2.circle(result, (int(p["x"]), int(p["y"])), int(p["radius"]),
                           (0, 0, 0), thickness=-1)
            elif tool == "put_black_rectangle":
                x, y = int(p["x"]), int(p["y"])
                cv2.rectangle(result, (x, y),
                              (x + int(p["w"]), y + int(p["h"])),
                              (0, 0, 0), thickness=-1)
        return result

    @staticmethod
    def _apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
        """Apply CLAHE to each channel of an RGB image independently."""
        import cv2

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        channels = [clahe.apply(image_rgb[:, :, c]) for c in range(3)]
        return np.stack(channels, axis=2)

    # =========================================================================
    # Private — PaDiM statistics
    # =========================================================================

    def _fit_gaussian(
        self,
        features: list[np.ndarray],
        params: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit per-pixel multivariate Gaussian using random feature projection.

        Args:
            features (list[np.ndarray]): Training feature maps
                ``(H', W', C')`` each.
            params (dict): Reads ``padim_dim`` (int) and ``padim_lambda``
                (float).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - ``mean``:       ``(H', W', D)`` float32 per-pixel mean.
                - ``precision``:  ``(H', W', D, D)`` float32 per-pixel
                  precision matrix.
                - ``random_idx``: ``(D,)`` int32 random column indices used for
                  projection.
        """
        padim_dim    = params.get("padim_dim",    100)
        padim_lambda = params.get("padim_lambda", 0.01)

        H, W, C = features[0].shape
        rng         = np.random.default_rng(seed=42)
        random_idx  = rng.choice(C, size=min(padim_dim, C), replace=False).astype(np.int32)

        # Stack all train images: (N, H, W, D)
        stacked = np.stack([f[:, :, random_idx] for f in features], axis=0).astype(np.float32)

        mean      = stacked.mean(axis=0)       # (H, W, D)
        centered  = stacked - mean             # (N, H, W, D)

        D = random_idx.shape[0]
        precision = np.zeros((H, W, D, D), dtype=np.float32)

        for i in range(H):
            for j in range(W):
                pixel_vecs = centered[:, i, j, :]   # (N, D)
                cov = (pixel_vecs.T @ pixel_vecs) / max(len(features) - 1, 1)
                cov += padim_lambda * np.eye(D, dtype=np.float32)
                precision[i, j] = np.linalg.inv(cov)

        return mean, precision, random_idx

    # =========================================================================
    # Private — scoring
    # =========================================================================

    def _score_all(
        self,
        features: list[np.ndarray],
        mean: np.ndarray,
        precision: np.ndarray,
        random_idx: np.ndarray,
        params: dict,
    ) -> np.ndarray:
        """
        Compute one anomaly score per image using the fitted Gaussian.

        Args:
            features (list[np.ndarray]): Feature maps ``(H', W', C')`` each.
            mean (np.ndarray): Fitted per-pixel mean ``(H', W', D)``.
            precision (np.ndarray): Per-pixel precision matrix ``(H', W', D, D)``.
            random_idx (np.ndarray): Random projection indices ``(D,)``.
            params (dict): Reads ``top_k_pixels`` (int), ``border_crop_px``
                (int), ``gaussian_sigma`` (float).

        Returns:
            np.ndarray: 1-D array of anomaly scores, one per image.
        """
        top_k        = params.get("top_k_pixels",   2)
        border_crop  = params.get("border_crop_px", 1)
        sigma        = params.get("gaussian_sigma", 0.0)

        scores: list[float] = []
        for feature_map in features:
            projected = feature_map[:, :, random_idx].astype(np.float32)
            error_map = self._mahalanobis_map(projected, mean, precision)

            if sigma > 0.0:
                from scipy.ndimage import gaussian_filter
                error_map = gaussian_filter(error_map, sigma=sigma)

            if border_crop > 0:
                error_map = error_map[border_crop:-border_crop,
                                      border_crop:-border_crop]

            flat = error_map.flatten()
            flat.sort()
            score = float(flat[-top_k:].mean()) if top_k <= len(flat) else float(flat.mean())
            scores.append(score)

        return np.array(scores, dtype=np.float32)

    def _extract_all_with_images(
        self,
        image_dir: str,
        roi: dict,
        training_shape: tuple[int, int],
        params: dict,
        masks: list[dict] | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """
        Like ``_extract_all`` but also returns the raw cropped+resized images
        for use in heatmap visualisation.

        Masks are applied to the full-resolution image BEFORE the ROI crop,
        matching the canonical order used at inference time.

        Args:
            image_dir (str): Directory of images.
            roi (dict): ``{x, y, w, h}`` crop in original capture coordinates.
            training_shape (tuple[int, int]): ``(height, width)`` to resize to.
            params (dict): Scoring params — reads ``use_clahe``.
            masks (list[dict] | None): Optional mask tool dicts applied before
                the ROI crop.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: ``(features, raw_images)``
                where each raw image is uint8 RGB at ``training_shape`` size,
                before ImageNet normalisation — suitable for display only.
        """
        import cv2

        use_clahe = params.get("use_clahe", False)
        x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
        target_h, target_w = training_shape
        features:   list[np.ndarray] = []
        raw_images: list[np.ndarray] = []

        if not os.path.isdir(image_dir):
            return features, raw_images

        for filename in sorted(os.listdir(image_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_bgr = cv2.imread(os.path.join(image_dir, filename))
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = self._apply_masks(img_rgb, masks or [])
            cropped = img_rgb[y:y + h, x:x + w]
            resized = cv2.resize(cropped, (target_w, target_h))
            raw_images.append(resized.copy())   # uint8 RGB — for visualisation only

            if use_clahe:
                resized = self._apply_clahe(resized)

            feature_map = self._extractor.extract_features(resized)
            features.append(feature_map)

        return features, raw_images

    def _score_with_error_maps(
        self,
        features: list[np.ndarray],
        mean: np.ndarray,
        precision: np.ndarray,
        random_idx: np.ndarray,
        params: dict,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Like ``_score_all`` but also returns the full (uncropped) error map per
        image for visualisation in heatmaps.

        Args:
            features (list[np.ndarray]): Feature maps ``(H', W', C')`` each.
            mean (np.ndarray): Fitted per-pixel mean ``(H', W', D)``.
            precision (np.ndarray): Per-pixel precision matrix ``(H', W', D, D)``.
            random_idx (np.ndarray): Random projection indices ``(D,)``.
            params (dict): Reads ``top_k_pixels``, ``border_crop_px``,
                ``gaussian_sigma``.

        Returns:
            tuple[np.ndarray, list[np.ndarray]]: ``(scores, error_maps)`` where
                each error map is ``(H', W')`` float32 before border cropping
                so the spatial layout is preserved for overlay.
        """
        top_k       = params.get("top_k_pixels",   2)
        border_crop = params.get("border_crop_px", 1)
        sigma       = params.get("gaussian_sigma", 0.0)

        scores:     list[float]      = []
        error_maps: list[np.ndarray] = []

        for feature_map in features:
            projected = feature_map[:, :, random_idx].astype(np.float32)
            error_map = self._mahalanobis_map(projected, mean, precision)

            if sigma > 0.0:
                from scipy.ndimage import gaussian_filter
                error_map = gaussian_filter(error_map, sigma=sigma)

            error_maps.append(error_map.copy())   # full map — for heatmap overlay

            if border_crop > 0:
                error_map_crop = error_map[border_crop:-border_crop,
                                           border_crop:-border_crop]
            else:
                error_map_crop = error_map

            flat = error_map_crop.flatten()
            flat.sort()
            score = float(flat[-top_k:].mean()) if top_k <= len(flat) else float(flat.mean())
            scores.append(score)

        return np.array(scores, dtype=np.float32), error_maps

    @staticmethod
    def _save_calibration_graphs(
        view_name: str,
        ok_scores: np.ndarray,
        nok_scores: np.ndarray,
        threshold_min: float,
        threshold_max: float,
        ok_images: list[np.ndarray],
        nok_images: list[np.ndarray],
        ok_error_maps: list[np.ndarray],
        nok_error_maps: list[np.ndarray],
        graphs_dir: str,
        timestamp: str,
    ) -> None:
        """
        Generate and save a score distribution histogram and per-image heatmaps.

        Two outputs are written to ``graphs_dir``:

        * ``{view_name}_score_dist_{timestamp}.png`` — histogram of OK and NOK
          score distributions with threshold lines and a metrics text box.
        * ``{view_name}_heatmap_{ok|nok}_{N}_{timestamp}.png`` — three-panel
          figure (original image / error map / overlay) for the first 5 OK
          images and ALL NOK images.

        Args:
            view_name (str): View identifier used in file names and titles.
            ok_scores (np.ndarray): 1-D scores for the test OK set.
            nok_scores (np.ndarray): 1-D scores for the test NOK set.
            threshold_min (float): Lower acceptance threshold.
            threshold_max (float): Upper acceptance threshold.
            ok_images (list[np.ndarray]): Raw uint8 RGB images for OK test set.
            nok_images (list[np.ndarray]): Raw uint8 RGB images for NOK test set.
            ok_error_maps (list[np.ndarray]): Error maps for OK test set.
            nok_error_maps (list[np.ndarray]): Error maps for NOK test set.
            graphs_dir (str): Output directory (created if absent).
            timestamp (str): Timestamp string appended to every file name.
        """
        import cv2
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe for background threads
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize

        os.makedirs(graphs_dir, exist_ok=True)

        # ── Score distribution histogram ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(14, 8))
        if len(ok_scores) > 0:
            ax.hist(ok_scores, bins=50, alpha=0.6, color="green",
                    label=f"OK ({len(ok_scores)})", density=True)
        if len(nok_scores) > 0:
            ax.hist(nok_scores, bins=50, alpha=0.6, color="red",
                    label=f"NOK ({len(nok_scores)})", density=True)
        ax.axvline(threshold_max, color="blue", linestyle="dashed", linewidth=2,
                   label=f"Max threshold ({threshold_max:.6f})")
        ax.axvline(threshold_min, color="purple", linestyle="dashed", linewidth=2,
                   label=f"Min threshold ({threshold_min:.6f})")
        ax.set_title(f"PaDiM Score Distribution — {view_name}")
        ax.set_xlabel("Anomaly Score (Mahalanobis distance)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        lines: list[str] = []
        if len(ok_scores) > 0:
            lines += [
                f"--- OK ({len(ok_scores)}) ---",
                f"  Min:  {float(np.min(ok_scores)):.6f}",
                f"  Max:  {float(np.max(ok_scores)):.6f}",
                f"  Mean: {float(np.mean(ok_scores)):.6f}",
                f"  Std:  {float(np.std(ok_scores)):.6f}",
            ]
        if len(nok_scores) > 0:
            lines += [
                f"--- NOK ({len(nok_scores)}) ---",
                f"  Min:  {float(np.min(nok_scores)):.6f}",
                f"  Max:  {float(np.max(nok_scores)):.6f}",
                f"  Mean: {float(np.mean(nok_scores)):.6f}",
                f"  Std:  {float(np.std(nok_scores)):.6f}",
            ]
        if len(ok_scores) > 0 and len(nok_scores) > 0:
            sep = float(np.min(nok_scores)) / (float(np.max(ok_scores)) + 1e-9)
            lines += [
                "--- Separation ---",
                f"  Ratio: {sep:.4f}x  (min(NOK)/max(OK))",
                "  GOOD — separated" if sep > 1.0 else "  WARNING — overlap",
            ]
        props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8)
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=8,
                verticalalignment="top", fontfamily="monospace", bbox=props)

        hist_path = os.path.join(graphs_dir, f"{view_name}_score_dist_{timestamp}.png")
        fig.savefig(hist_path, bbox_inches="tight", dpi=100)
        plt.close(fig)
        print(f"[OK] Graph saved: {hist_path}")

        # ── Heatmaps — first 5 OK + all NOK ──────────────────────────────────
        entries: list[tuple[np.ndarray, np.ndarray, float, str]] = []
        for i, (img, emap, score) in enumerate(
                zip(ok_images, ok_error_maps, ok_scores.tolist())):
            if i >= 5:
                break
            entries.append((img, emap, float(score), f"OK #{i + 1}"))
        for i, (img, emap, score) in enumerate(
                zip(nok_images, nok_error_maps, nok_scores.tolist())):
            entries.append((img, emap, float(score), f"NOK #{i + 1}"))

        if not entries:
            return

        all_emaps = [e[1] for e in entries]
        global_max = max(float(e.max()) for e in all_emaps) or 1.0
        norm       = Normalize(vmin=0.0, vmax=global_max)
        colormap   = matplotlib.colormaps.get_cmap("jet")

        for img, emap, score, label in entries:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Panel 1 — original image (uint8 RGB)
            axes[0].imshow(img)
            axes[0].set_title("Input image")
            axes[0].axis("off")

            # Panel 2 — error map (jet colourmap)
            hm = axes[1].imshow(emap, cmap="jet", norm=norm)
            axes[1].set_title(f"Error map  (score={score:.4f})")
            axes[1].axis("off")
            fig.colorbar(hm, ax=axes[1], fraction=0.046, pad=0.04)

            # Panel 3 — overlay: error map blended onto original
            emap_resized = cv2.resize(emap, (img.shape[1], img.shape[0]))
            rgba         = colormap(norm(emap_resized))
            heatmap_rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
            overlay      = cv2.addWeighted(img, 0.55, heatmap_rgb, 0.45, 0)
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay (55% image + 45% heatmap)")
            axes[2].axis("off")

            ok_or_nok = "ok" if label.startswith("OK") else "nok"
            idx_str   = label.split("#")[1].strip().zfill(2)
            fig.suptitle(
                f"{view_name} — {label}   score={score:.4f}"
                f"  [threshold: {threshold_min:.4f}–{threshold_max:.4f}]",
                fontsize=11,
            )
            fig.tight_layout()
            hm_path = os.path.join(
                graphs_dir,
                f"{view_name}_heatmap_{ok_or_nok}_{idx_str}_{timestamp}.png",
            )
            fig.savefig(hm_path, bbox_inches="tight", dpi=100)
            plt.close(fig)

        print(f"[OK] Heatmaps saved to: {graphs_dir}")

    @staticmethod
    def _mahalanobis_map(
        projected: np.ndarray,
        mean: np.ndarray,
        precision: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a 2-D Mahalanobis distance map.

        Args:
            projected (np.ndarray): ``(H', W', D)`` projected feature map.
            mean (np.ndarray): ``(H', W', D)`` per-pixel mean.
            precision (np.ndarray): ``(H', W', D, D)`` precision matrices.

        Returns:
            np.ndarray: ``(H', W')`` float32 distance map.
        """
        diff    = projected - mean                 # (H, W, D)
        # dist[i,j] = diff[i,j] @ precision[i,j] @ diff[i,j]
        tmp     = np.einsum("hwi,hwij->hwj", diff, precision)  # (H, W, D)
        dist_sq = np.einsum("hwi,hwi->hw",  tmp,  diff)        # (H, W)
        return np.sqrt(np.maximum(dist_sq, 0.0))

    # =========================================================================
    # Private — metrics
    # =========================================================================

    @staticmethod
    def _compute_auc(ok_scores: np.ndarray, nok_scores: np.ndarray) -> float:
        """
        Compute the ROC-AUC between OK and NOK score distributions.

        Args:
            ok_scores (np.ndarray): Anomaly scores for test OK images.
            nok_scores (np.ndarray): Anomaly scores for test NOK images.

        Returns:
            float: AUC value in ``[0.0, 1.0]``.
        """
        from sklearn.metrics import roc_auc_score

        labels = np.concatenate([
            np.zeros(len(ok_scores),  dtype=np.int32),
            np.ones( len(nok_scores), dtype=np.int32),
        ])
        scores = np.concatenate([ok_scores, nok_scores])
        return float(roc_auc_score(labels, scores))

    @staticmethod
    def _build_sweep_result(
        view_name: str,
        block_results: list[BlockSweepResult],
        min_sep_gate: float = _DEFAULT_MIN_SEP_GATE,
    ) -> SweepResult:
        """
        Build a ``SweepResult`` by selecting the best block from block results.

        Selection rule: among the blocks whose ``sep_ratio`` is strictly
        greater than ``min_sep_gate`` (i.e. NOK images are actually
        separated from OK images), pick the one with the lowest ``cv_ok``
        — the most consistent/stable OK score distribution. This favors
        calibration stability over raw separation once separation is already
        adequate.

        If no block passes the gate (e.g. no NOK images were available, or
        none reach real separation), falls back to the block with the
        highest ``sep_ratio``, matching the previous behavior.

        Args:
            view_name (str): The view this sweep was run for.
            block_results (list[BlockSweepResult]): One entry per block evaluated.
            min_sep_gate (float): Minimum ``sep_ratio`` a block must exceed to
                be eligible for the ``cv_ok``-based selection.

        Returns:
            SweepResult: Aggregated result with the selected ``best_block``.
        """
        if not block_results:
            return SweepResult(view_name=view_name)

        candidates = [r for r in block_results if r.sep_ratio > min_sep_gate]
        best = min(candidates, key=lambda r: r.cv_ok) if candidates else \
            max(block_results, key=lambda r: r.sep_ratio)

        return SweepResult(
            view_name=view_name,
            block_results=block_results,
            best_block=best.block,
            best_sep_ratio=best.sep_ratio,
            best_auc=best.auc,
            best_cv_ok=best.cv_ok,
        )
