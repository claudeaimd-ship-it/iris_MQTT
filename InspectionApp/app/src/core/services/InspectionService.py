import numpy as np

from app.src.core.models.Part import Part, InspectionResult
from app.src.core.settings.SequenceSettings import SequenceSettings
from app.src.interfaces.ICamera import ICamera
from app.src.interfaces.IInferenceEngine import IInferenceEngine
from app.src.interfaces.IRepository import IRepository

class InspectionService:
    def __init__(
        self,
        icamera: ICamera,
        iinference_engine: IInferenceEngine,
        irepository: IRepository,
        settings: SequenceSettings,
    ):
        """
        Args:
            icamera: Camera interface for capturing images.
            iinference_engine: Inference engine interface for running model predictions.
            irepository: Repository interface for data storage and retrieval.
            settings (SequenceSettings): Configuration wrapper providing preprocessing
                pipelines and thresholds for each view.
        """
        self._camera = icamera
        self._inference_engine = iinference_engine
        self._repository = irepository
        self._settings = settings

    def execute_full_inspection_from_frames(self, part: Part, captured_frames: dict) -> None:
        """
        Run inference on a set of already-captured frames and populate the Part.

        Called by SequenceExecutor after all camera steps have completed. The
        captured_frames dict uses view names as keys (e.g. 'front_view_section_1_A')
        which encode both the view prefix and the camera channel.

        Args:
            part (Part): Part instance to populate with InspectionResult entries.
            captured_frames (dict[str, np.ndarray]): Frames keyed by view name,
                as collected during the capture steps of the sequence.

        Returns:
            None
        """
        for view_name, image in captured_frames.items():
            # View name format: '{prefix}_{channel}', e.g. 'front_view_section_1_A'.
            # The last segment is always the camera channel.
            channel = view_name.split("_")[-1]
            section = "_".join(view_name.split("_")[:-1])

            threshold_min, threshold_max = self._settings.get_threshold_for_view(view_name)
            score, error_map = self._run_inference_on_view(image, view_name, section, channel)
            is_ok = threshold_min <= score <= threshold_max

            # Presence-detection views calibrate on the ABSENT feature (baseline);
            # an anomaly (score outside range) means the feature is present, which is OK here.
            if self._settings.get_inference_type(view_name) == "presence_detection_absence_calibrated":
                is_ok = not is_ok

            result = InspectionResult(
                view=view_name,
                is_ok=is_ok,
                score=score,
                threshold_used=(threshold_min, threshold_max),
                error_map=error_map,
            )
            part.inspection_results.append(result)

        #self._repository.save_inspection_result(part)
        #self._repository.save_frames(part.part_id, captured_frames)

    def save_results(self, part: Part, captured_frames: dict) -> None:
        """
        Save the inspection results and captured frames to the repository.

        This is separated from the inference logic to allow for flexibility in
        when and how results are persisted (e.g., after each part, in batches, etc.).

        Only frames for NOK views are persisted (per-view filtering, not
        per-part) — OK inference frames are never used by the downstream
        review/promote pipeline (``TraceabilityReviewService``, the review
        page, "Recalibrate from Production"), and persisting every OK frame
        of every cycle causes unnecessary SD-card wear. This does not affect
        ``images_path/latest/`` (the live-preview fallback), which is written
        by a separate mechanism and always keeps saving every cycle.

        Args:
            part (Part): Part instance containing populated inspection results.
            captured_frames (dict[str, np.ndarray]): Frames keyed by view name.
        Returns:
            None
        """
        nok_views = {r.view for r in part.inspection_results if not r.is_ok}
        frames_to_save = {
            view_name: image
            for view_name, image in captured_frames.items()
            if view_name in nok_views
        }
        self._repository.save_inspection_result(part)
        self._repository.save_frames(part.part_id, frames_to_save)

    def _run_inference_on_view(
        self,
        frame: np.ndarray,
        view_name: str,
        section: str,
        channel: str,
    ) -> tuple[float, np.ndarray]:
        """
        Preprocess a captured frame and run split inference on it.

        Preprocessing is fully defined by the JSON pipeline for this view
        (SequenceSettings). The teacher extracts features; the student reconstructs
        them. The anomaly score is computed on the feature-space difference.

        Args:
            frame (np.ndarray): Raw captured image from the camera.
            view_name (str): Full view key, e.g. ``front_view_section_1_A``.
            section (str): Section component of view_name, e.g. ``front_view_section_1``.
            channel (str): Channel component of view_name, e.g. ``A``.

        Returns:
            tuple[float, np.ndarray]: ``(score, error_map_2d)``.
        """
        preprocessed = self._settings.preprocess_for_inference(frame, view_name)
        features_original, features_reconstructed = self._inference_engine.predict(
            preprocessed, section, channel
        )
        return self._calculate_anomaly_score_and_error_map(
            features_original, features_reconstructed
        )

    def _calculate_anomaly_score_and_error_map(
        self,
        features_original: np.ndarray,
        features_reconstructed: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute the anomaly score and a 2-D error map from teacher/student feature maps.

        Mirrors the scoring logic from ``training_pytorch_float.py``:
        - MSE averaged over the channel axis → 2-D error map.
        - Crop 1-pixel border to remove convolution edge artifacts.
        - Score = mean of the top-k pixel errors (k = ``settings.top_k_pixels``).

        Feature maps from TFLite are in NHWC format: ``(1, H', W', C')``.

        Args:
            features_original (np.ndarray): Original features from the teacher model.
            features_reconstructed (np.ndarray): Reconstructed features from the student.

        Returns:
            tuple[float, np.ndarray]: ``(score, error_map_2d)`` where ``error_map_2d``
                has shape ``(H'-2, W'-2)`` after border cropping.
        """
        diff_sq = (
            features_original.astype(np.float32)
            - features_reconstructed.astype(np.float32)
        ) ** 2

        # Average squared error over the channel axis.
        # TFLite NHWC: axes are (batch, H, W, C) → average over axis=-1, take batch 0.
        if diff_sq.ndim == 4:      # (N, H, W, C)
            error_map = diff_sq[0].mean(axis=-1)   # (H, W)
        elif diff_sq.ndim == 3:    # (H, W, C)
            error_map = diff_sq.mean(axis=-1)      # (H, W)
        else:
            error_map = diff_sq

        # Optional Gaussian smoothing (sigma > 0 must match training configuration).
        sigma = self._settings.gaussian_sigma
        if sigma > 0.0:
            from scipy import ndimage  # Optional dependency; only imported when needed.
            error_map = ndimage.gaussian_filter(error_map, sigma=sigma)

        # Crop border to remove edge convolution artifacts.
        crop = self._settings.border_crop_px
        if crop > 0 and error_map.shape[0] > 2 * crop and error_map.shape[1] > 2 * crop:
            error_map = error_map[crop:-crop, crop:-crop]

        # Top-k average score — same formula as training.
        k    = self._settings.top_k_pixels
        flat = error_map.flatten()
        score = float(np.mean(np.sort(flat)[-k:])) if len(flat) >= k else float(np.mean(flat))

        return score, error_map