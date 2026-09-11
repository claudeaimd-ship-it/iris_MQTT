import numpy as np

from app.src.core.utils.ImageProcessing import (
    apply_clahe,
    apply_roi_crop,
    put_black_circle,
    put_black_rectangle,
    resize_to_training_resolution,
    normalize_for_mobilenet,
    add_batch_dimension,
)


class SequenceSettings:
    """
    Wraps the sequence JSON and the thresholds dict to provide typed access to
    view-specific preprocessing pipelines and inference thresholds.

    Replaces the role of UtilidadesTFLite (coordinate lookup + preprocessing)
    and AdministradorArchivos (threshold loading) from the previous architecture.

    The preprocessing pipeline for each view is defined in the sequence JSON under
    ``preprocessing_image_parameters``. Each entry specifies a ``view``, ``section``,
    optional ``camera_port``, and a ``pipeline`` list of tool dicts applied in order.

    Attributes:
        _pipelines (dict[str, list[dict]]): Preprocessing pipeline per view_name.
        _thresholds (dict[str, dict]): Min/max thresholds per view_name.
        _top_k_pixels (int): Number of top-error pixels used for anomaly scoring,
            matching the ``k_pixeles`` parameter from training_pytorch_float.py.

    Note:
        view_name is always ``{prefix_view}_{channel}``, e.g. ``front_view_section_1_A``.
        It must match the keys used in ``captured_frames`` by SequenceExecutor.
    """

    DEFAULT_TOP_K_PIXELS  = 15   # Matches k_pixeles=15 from training_pytorch_float.py.
    DEFAULT_BORDER_CROP_PX = 1   # 1-pixel border crop removes edge convolution artifacts.
    DEFAULT_GAUSSIAN_SIGMA = 0.0 # 0.0 = disabled; set > 0 in eval_config to apply smoothing.

    # Canonical tool execution order — independent of JSON order.
    # Masks (phase 1) must run on the full-resolution image before the ROI crop (phase 2).
    # Unknown tools default to phase 50 (after resize, before normalize).
    _TOOL_ORDER: dict[str, int] = {
        "apply_clahe":                   0,
        "put_black_circle":              1,
        "put_black_rectangle":           1,
        "apply_roi_crop":                2,
        "resize_to_training_resolution": 3,
        "normalize_mobilenet":           4,
        "set_time_exposure":             99,
        "set_lens_position":             99,
    }

    def __init__(self, sequence: dict, thresholds: dict, eval_config: dict | None = None):
        """
        Args:
            sequence (dict): Parsed sequence JSON dict. Must contain
                ``preprocessing_image_parameters``.
            thresholds (dict): Dict loaded from ``thresholds.json`` in the models
                directory. Expected format::

                    {
                        "front_view_section_1_A": {"min": 0.0001, "max": 0.0050},
                        ...
                    }
            eval_config (dict | None): Model-level scoring parameters loaded from
                ``eval_config.json`` alongside the model files.  Takes precedence
                over values from the ``scoring`` block of the sequence JSON.
                Expected keys: ``top_k_pixels``, ``border_crop_px``,
                ``gaussian_sigma``.  Any missing key falls back to sequence JSON
                or built-in defaults.
        """
        eval_cfg = eval_config or {}
        self._thresholds = thresholds

        # Scoring parameters — priority: eval_config > sequence JSON > defaults.
        scoring = sequence.get("scoring", {})
        self._top_k_pixels: int    = eval_cfg.get(
            "top_k_pixels",  scoring.get("top_k_pixels",  self.DEFAULT_TOP_K_PIXELS)
        )
        self._border_crop_px: int  = eval_cfg.get(
            "border_crop_px", scoring.get("border_crop_px", self.DEFAULT_BORDER_CROP_PX)
        )
        self._gaussian_sigma: float = eval_cfg.get(
            "gaussian_sigma", scoring.get("gaussian_sigma", self.DEFAULT_GAUSSIAN_SIGMA)
        )

        # Build lookup: view_name -> pipeline list.
        # Key is constructed from view + section + camera_port fields in the JSON entry.
        # Entries without camera_port serve as fallback defaults for that view/section.
        self._pipelines: dict[str, list[dict]] = {}
        self._port_pipelines: dict[str, list[dict]] = {}   # fallback: camera_port -> pipeline
        # Same key scheme as _pipelines/_port_pipelines — see get_inference_type().
        self._inference_types: dict[str, str] = {}
        self._port_inference_types: dict[str, str] = {}
        for entry in sequence.get("preprocessing_image_parameters", []):
            view        = entry.get("view", "")
            section     = entry.get("section", "")
            camera_port = entry.get("camera_port", None)
            pipeline    = entry.get("pipeline", [])
            inference_type = entry.get("inference_type", "standard")

            prefix = view

            key = f"{prefix}_{camera_port}" if camera_port else prefix
            self._pipelines[key] = pipeline
            self._inference_types[key] = inference_type

            if camera_port:
                self._port_pipelines[camera_port] = pipeline
                self._port_inference_types[camera_port] = inference_type

    # =========================================================================
    # Public API
    # =========================================================================

    def preprocess_for_inference(self, frame: np.ndarray, view_name: str) -> np.ndarray:
        """
        Apply the preprocessing pipeline defined for this view in the sequence JSON.

        Lookup order:
        1. Exact ``view_name`` match (e.g. ``front_view_section_1_A``).
        2. Prefix match without the channel suffix (e.g. ``front_view_section_1``).
        3. Port match — finds the pipeline entry whose ``camera_port`` equals the
           channel suffix of ``view_name``.  Handles the case where ``prefix_view``
           in a camera_action step differs from the ``view`` field in
           ``preprocessing_image_parameters`` (e.g. ``prefix_view="a"`` vs
           ``view="section_view"`` for the same camera port).
        4. Empty pipeline — frame is returned with only a batch dimension added.

        Supported pipeline tools:
        - ``apply_clahe``: CLAHE on each RGB channel.
        - ``apply_roi_crop``: Crop a rectangle. Params: ``x, y, w, h``.
        - ``put_black_circle``: Mask a circle. Params: ``x, y, radius``.
        - ``put_black_rectangle``: Mask a rectangle. Params: ``x, y, h, w``.
        - ``resize_to_training_resolution``: Resize. Params: ``width, height``.
        - ``normalize_mobilenet``: MobileNetV2 ImageNet normalization.
        - ``set_time_exposure``, ``set_lens_position``: Camera controls already
          handled by SequenceExecutor; silently skipped here.

        Args:
            frame (np.ndarray): Raw captured frame from the camera.
            view_name (str): Key matching the captured_frames dict, e.g.
                ``front_view_section_1_A``.

        Returns:
            np.ndarray: Preprocessed image with batch dimension, shape (1, H, W, C),
                ready to be fed to a TFLite interpreter.
        """
        pipeline = self._pipelines.get(view_name)

        if pipeline is None:
            # Fallback 2: strip the channel suffix and try the prefix.
            prefix   = "_".join(view_name.split("_")[:-1])
            pipeline = self._pipelines.get(prefix)

        if pipeline is None:
            # Fallback 3: match by camera_port alone (handles prefix_view ≠ view mismatch).
            channel  = view_name.split("_")[-1]
            pipeline = self._port_pipelines.get(channel, [])

        result = frame.copy()
        ordered = sorted(pipeline, key=lambda t: self._TOOL_ORDER.get(t.get("tool", ""), 50))
        for tool_def in ordered:
            tool   = tool_def.get("tool", "")
            params = tool_def.get("parameters", {})

            if tool == "apply_clahe":
                result = apply_clahe(result)

            elif tool == "apply_roi_crop":
                result = apply_roi_crop(
                    result, params["x"], params["y"], params["w"], params["h"]
                )
            elif tool == "put_black_circle":
                result = put_black_circle(
                    result, params["x"], params["y"], params["radius"]
                )
            elif tool == "put_black_rectangle":
                result = put_black_rectangle(
                    result, params["x"], params["y"], params["w"], params["h"]
                )
            elif tool == "resize_to_training_resolution":
                result = resize_to_training_resolution(
                    result, params["width"], params["height"]
                )
            elif tool == "normalize_mobilenet":
                result = normalize_for_mobilenet(result)

            elif tool in ("set_time_exposure", "set_lens_position"):
                pass  # Camera-control steps handled by SequenceExecutor; skip here.

            else:
                print(f"[WARN] SequenceSettings: unknown pipeline tool '{tool}' — skipped.")

        return add_batch_dimension(result)

    def get_threshold_for_view(self, view_name: str) -> tuple[float, float]:
        """
        Return the (min, max) acceptance threshold for the given view.

        Args:
            view_name (str): Key identifying the view, e.g. ``front_view_section_1_A``.

        Returns:
            tuple[float, float]: ``(threshold_min, threshold_max)``.

        Raises:
            KeyError: If ``thresholds.json`` does not contain an entry for ``view_name``.
        """
        entry = self._thresholds.get(view_name)
        if entry is None:
            raise KeyError(
                f"No threshold found for view '{view_name}'. "
                f"Ensure thresholds.json in the models directory contains this key."
            )
        return float(entry["min"]), float(entry["max"])

    def get_inference_type(self, view_name: str) -> str:
        """
        Return the inference type configured for the given view.

        Same 3-level fallback lookup as ``preprocess_for_inference()``: exact
        ``view_name`` match, then prefix without the channel suffix, then
        ``camera_port`` alone.

        Args:
            view_name (str): Key identifying the view, e.g. ``front_view_section_1_A``.

        Returns:
            str: ``"standard"`` (default) or ``"presence_detection_absence_calibrated"``.
        """
        inference_type = self._inference_types.get(view_name)

        if inference_type is None:
            prefix = "_".join(view_name.split("_")[:-1])
            inference_type = self._inference_types.get(prefix)

        if inference_type is None:
            channel = view_name.split("_")[-1]
            inference_type = self._port_inference_types.get(channel, "standard")

        return inference_type

    @property
    def top_k_pixels(self) -> int:
        """Number of top-error pixels used for anomaly scoring."""
        return self._top_k_pixels

    @property
    def border_crop_px(self) -> int:
        """Pixels cropped from each border of the error map to remove edge artifacts."""
        return self._border_crop_px

    @property
    def gaussian_sigma(self) -> float:
        """
        Sigma for optional Gaussian blur applied to the error map before scoring.

        A value of ``0.0`` (the default) disables smoothing entirely.
        Requires ``scipy`` to be installed when set to a positive value.
        """
        return self._gaussian_sigma
