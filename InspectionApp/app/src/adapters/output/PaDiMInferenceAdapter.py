import json
import os

import numpy as np
import onnxruntime as ort

from app.src.interfaces.IInferenceEngine import IInferenceEngine


class PaDiMInferenceAdapter(IInferenceEngine):
    """
    Inference adapter using PaDiM (Patch Distribution Modeling) for anomaly detection.

    Unlike the Teacher-Student split model, PaDiM uses a single frozen backbone
    to extract patch features and compares them to a multivariate Gaussian
    (mean + precision matrix) fitted per spatial position during calibration.
    No backpropagation and no GPU are required — calibration runs entirely on CPU,
    making it suitable for the Raspberry Pi 4B.

    The anomaly score is the Mahalanobis distance between the extracted patch
    features and the calibrated Gaussian at each spatial position.

    **Compatibility with** ``InspectionService._calculate_anomaly_score_and_error_map``:

    ``predict()`` returns ``(sqrt(d_map), zeros)`` both of shape ``(1, H', W', 1)``.
    ``InspectionService`` then computes ``(A - B)^2.mean(axis=-1) = d_map``,
    which is the raw Mahalanobis distance map — identical to what the training
    script's ``compute_error_map()`` returns.  The downstream blur + border-crop
    + top-k scoring pipeline is therefore applied on the same values.

    Model files required in ``model_path``::

        padim_{view_name}_params.npz  — calibration params: mean, precision, random_idx
        eval_config.json              — scoring params + backbone_paths per view

    ``eval_config.json`` must contain a ``backbone_paths`` dict mapping each
    view_name to the ONNX backbone path (relative to ``model_path``).  This is
    written automatically by ``CalibrationService.run_calibration()``.

    Attributes:
        _views (dict[str, tuple]): Per-view data keyed by ``view_name``.
            Each value is ``(ort_session, mean, precision, random_idx)``.
        _eval_config (dict): Scoring parameters loaded from ``eval_config.json``.
    """

    _PARAMS_PREFIX        = "padim_"
    _PARAMS_SUFFIX        = "_params.npz"
    _EVAL_CONFIG_FILENAME = "eval_config.json"

    def __init__(self):
        self._views: dict[str, tuple] = {}
        self._eval_config: dict = {}

    # =========================================================================
    # IInferenceEngine interface
    # =========================================================================

    def load_models_from_directory(self, models_dir: str) -> None:
        """
        Scan ``models_dir`` for all PaDiM params files and load them.

        ``eval_config.json`` must be present and contain a ``backbone_paths``
        dict mapping each view_name to the ONNX backbone path (relative to
        ``models_dir``).  Written automatically by ``CalibrationService``.

        Args:
            models_dir (str): Directory containing the model files.

        Raises:
            FileNotFoundError: If ``models_dir`` does not exist.
        """
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f"Models directory not found: '{models_dir}'")

        # Read eval_config.json first — needed for backbone_paths.
        eval_config_path = os.path.join(models_dir, self._EVAL_CONFIG_FILENAME)
        if os.path.isfile(eval_config_path):
            with open(eval_config_path, "r") as f:
                self._eval_config = json.load(f)
            print(f"[OK] eval_config.json loaded from '{models_dir}'.")
        else:
            print(
                f"[WARN] eval_config.json not found in '{models_dir}'. "
                f"Backbone paths cannot be resolved — inference will fail."
            )
            self._eval_config = {}

        backbone_paths: dict[str, str] = self._eval_config.get("backbone_paths", {})

        params_files = [
            f for f in os.listdir(models_dir)
            if f.startswith(self._PARAMS_PREFIX) and f.endswith(self._PARAMS_SUFFIX)
        ]

        loaded = 0
        for params_filename in params_files:
            view_name = params_filename[
                len(self._PARAMS_PREFIX) : -len(self._PARAMS_SUFFIX)
            ]

            params_path = os.path.join(models_dir, params_filename)

            backbone_rel = backbone_paths.get(view_name)
            if not backbone_rel:
                print(
                    f"[WARN] PaDiMInferenceAdapter: no backbone_path in eval_config for "
                    f"view '{view_name}' — skipped. Re-run calibration to update eval_config.json."
                )
                continue

            backbone_path = os.path.normpath(os.path.join(models_dir, backbone_rel))
            if not os.path.isfile(backbone_path):
                print(
                    f"[WARN] PaDiMInferenceAdapter: backbone not found at '{backbone_path}' "
                    f"for view '{view_name}' — skipped."
                )
                continue

            data       = np.load(params_path)
            mean       = data["mean"]       # (H', W', D)
            precision  = data["precision"]  # (H', W', D, D)
            random_idx = data["random_idx"] # (D,)

            sess = ort.InferenceSession(
                backbone_path,
                providers=["CPUExecutionProvider"],
            )

            self._views[view_name] = (sess, mean, precision, random_idx)
            loaded += 1
            print(f"[OK] PaDiMInferenceAdapter: loaded params for view '{view_name}'")

        if loaded == 0:
            print(
                f"[WARN] PaDiMInferenceAdapter: no PaDiM parameter files loaded from "
                f"'{models_dir}'. Inference will fail until calibration is run."
            )
            return

        print(f"[OK] PaDiMInferenceAdapter: {loaded} view(s) loaded from '{models_dir}'.")

    @property
    def eval_config(self) -> dict:
        """
        Model-level scoring parameters loaded from ``eval_config.json``.

        Returns:
            dict: Parsed ``eval_config.json`` content, or ``{}`` if not found.
        """
        return self._eval_config

    def predict(
        self,
        input_data: np.ndarray,
        section: str,
        channel: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Mahalanobis distance map and return it in Teacher-Student format.

        Passes ``input_data`` through the frozen backbone to get patch features,
        reduces to the calibrated random dimensions, then computes the Mahalanobis
        distance at each spatial position using the stored mean and precision matrix.

        **Return convention**: returns ``(sqrt(d_map), zeros)`` so that
        ``InspectionService._calculate_anomaly_score_and_error_map`` computes
        ``(A - B)^2.mean(axis=-1) = d_map``, matching the training script exactly.

        Args:
            input_data (np.ndarray): Preprocessed image, shape ``(1, H, W, 3)``,
                uint8 or float32 in ``[0, 255]`` range — as returned by
                ``SequenceSettings.preprocess_for_inference``.  MobileNetV2
                normalisation is applied internally before the backbone forward
                pass, so the pipeline JSON must NOT include ``normalize_mobilenet``
                for PaDiM sequences (unlike Teacher-Student where normalisation is
                baked into the TFLite export).
            section (str): View section component of the view_name,
                e.g. ``primera_mitad``.
            channel (str): Camera channel component of the view_name, e.g. ``C``.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(sqrt_d_map_nhwc, zeros_nhwc)``,
                both float32 with shape ``(1, H', W', 1)``.

        Raises:
            KeyError: If no PaDiM parameters were loaded for this view_name.
        """
        view_name = f"{section}_{channel}"

        if view_name not in self._views:
            raise KeyError(
                f"No PaDiM parameters loaded for view '{view_name}'. "
                f"Loaded views: {list(self._views.keys())}"
            )

        sess, mean, precision, random_idx = self._views[view_name]

        # ── Step 1: extract backbone features ──────────────────────────────
        # Normalise to MobileNetV2 ImageNet statistics, matching the preprocessing
        # applied by PaDiMFeatureExtractorAdapter._preprocess during calibration.
        # The pipeline JSON does NOT include normalize_mobilenet for PaDiM sequences
        # (unlike Teacher-Student where normalisation is baked into the TFLite model),
        # so normalisation must be done here unconditionally.
        backbone_input = self._normalize_mobilenet(input_data)

        input_name = sess.get_inputs()[0].name
        features = sess.run(None, {input_name: backbone_input})[0]
        # features shape: (1, H', W', C) — NHWC from calibration backbone

        features = features[0]         # (H', W', C)
        Hp, Wp, _ = features.shape
        D = len(random_idx)

        # ── Step 2: reduce to calibrated random dimensions ─────────────────
        feat = features[:, :, random_idx]   # (H', W', D)
        diff = (feat - mean).reshape(-1, D) # (P, D)
        prec = precision.reshape(-1, D, D)  # (P, D, D)

        # ── Step 3: Mahalanobis distance per spatial position ───────────────
        # d²[k] = diff[k] @ precision[k] @ diff[k]ᵀ
        intermediate = np.einsum("pi,pij->pj", diff, prec)   # (P, D)
        dist2        = np.einsum("pi,pi->p",   intermediate, diff)  # (P,)
        dist2        = np.maximum(dist2, 0.0)                        # numerical safety
        d_map        = np.sqrt(dist2).reshape(Hp, Wp).astype(np.float32)  # (H', W')

        # ── Step 4: pack as (sqrt(d_map), zeros) for InspectionService ─────
        # InspectionService computes: (A - B)² .mean(axis=-1)
        #   = (sqrt(d_map) - 0)² .mean(axis=-1)
        #   = d_map[:, :, 0]          (since last dim is 1, mean is identity)
        #   = Mahalanobis distance map   ✓
        sqrt_d = np.sqrt(d_map)[np.newaxis, :, :, np.newaxis]  # (1, H', W', 1)
        zeros  = np.zeros((1, Hp, Wp, 1), dtype=np.float32)

        return sqrt_d, zeros

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _normalize_mobilenet(input_data: np.ndarray) -> np.ndarray:
        """
        Normalise a batch image to MobileNetV2 ImageNet statistics.

        Mirrors ``PaDiMFeatureExtractorAdapter._preprocess`` exactly, ensuring
        that the backbone ONNX receives the same input distribution at inference
        time as it did during calibration.

        Args:
            input_data (np.ndarray): ``(1, H, W, 3)`` image as produced by
                ``SequenceSettings.preprocess_for_inference``.  May be uint8 or
                float32 — values in ``[0, 255]`` are rescaled to ``[0.0, 1.0]``.

        Returns:
            np.ndarray: ``(1, H, W, 3)`` float32 normalised tensor.
        """
        img = input_data.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (img - mean) / std
