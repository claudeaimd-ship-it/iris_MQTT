import numpy as np

from abc import ABC, abstractmethod


class IInferenceEngine(ABC):
    """
    Abstract interface for inference engine operations.

    The system uses a split-inference architecture:
    - A teacher model (MobileNetV2 float32) extracts features from the input image.
    - A student autoencoder (int8, EdgeTPU) reconstructs those features.
    - The anomaly score is derived from the difference between the two feature maps.

    Implementations must load all model pairs from a directory at startup
    and select the correct pair by (section, channel) at prediction time.
    """

    @abstractmethod
    def load_models_from_directory(self, models_dir: str) -> None:
        """
        Scan ``models_dir`` for all teacher/student model file pairs and load them.

        File naming convention::

            teacher_{view_name}_float32.tflite      ← teacher (CPU, float32)
            student_{view_name}_int8_edgetpu.tflite   ← student (EdgeTPU, int8)

        Where ``view_name`` matches the keys used in ``captured_frames``,
        e.g. ``front_view_section_1_A``.

        Args:
            models_dir (str): Path to the directory containing the model files.

        Raises:
            FileNotFoundError: If ``models_dir`` does not exist.
            RuntimeError: If no valid model pairs are found in the directory.
        """
        pass

    @abstractmethod
    def predict(
        self,
        input_data: np.ndarray,
        section: str,
        channel: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run split inference for the specified view.

        Passes ``input_data`` through the teacher model to extract features, then
        through the student model to reconstruct them. Both feature maps are returned
        so the caller (InspectionService) can compute the anomaly score.

        Args:
            input_data (np.ndarray): Preprocessed image with batch dimension,
                shape ``(1, H, W, C)``, float32 normalized for MobileNetV2.
            section (str): View section component of the view_name,
                e.g. ``front_view_section_1``.
            channel (str): Camera channel component of the view_name, e.g. ``A``.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(features_original, features_reconstructed)``
                both as float32 arrays with the same shape, e.g. ``(1, H', W', C')``.

        Raises:
            KeyError: If no model pair is loaded for the given (section, channel).
        """
        pass

