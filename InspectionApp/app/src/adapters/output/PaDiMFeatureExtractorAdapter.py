import numpy as np

from app.src.interfaces.IPaDiMFeatureExtractor import IPaDiMFeatureExtractor


class PaDiMFeatureExtractorAdapter(IPaDiMFeatureExtractor):
    """
    PaDiM backbone feature extractor using ONNX Runtime.

    Loads a pre-exported MobileNetV2 backbone (``.onnx``) and runs it on a
    single RGB image to produce a spatial feature map. Used exclusively during
    calibration (sweep and fit); inference uses the fitted
    ``padim_*_params.npz`` files directly.

    The backbone files are platform-agnostic — the same ``.onnx`` files work
    on PC (development) and on the Raspberry Pi.

    Note:
        ONNX Runtime and numpy must be installed. On RPi, also install
        ``scikit-learn`` and ``scipy`` for ``CalibrationService``.

    Attributes:
        _session (onnxruntime.InferenceSession | None): ONNX inference session.
            None until ``load_backbone()`` is called.
        _input_name (str | None): Name of the ONNX graph input node.
    """

    def __init__(self) -> None:
        self._session    = None
        self._input_name: str | None = None

    # =========================================================================
    # IPaDiMFeatureExtractor
    # =========================================================================

    def load_backbone(self, backbone_path: str) -> None:
        """
        Load the ONNX backbone for a specific MobileNetV2 block.

        Args:
            backbone_path (str): Absolute path to the ``.onnx`` backbone file
                (e.g. ``data/models/backbones/teacher_mobilenetv2_backbone_b9.onnx``).

        Returns:
            None

        Raises:
            FileNotFoundError: If the backbone file does not exist.
            RuntimeError: If ONNX Runtime fails to load the session.
        """
        import onnxruntime as ort

        self._session    = ort.InferenceSession(backbone_path)
        self._input_name = self._session.get_inputs()[0].name
        print(f"[OK] PaDiMFeatureExtractorAdapter: backbone loaded from '{backbone_path}'.")

    def extract_features(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Extract a spatial feature map from a single RGB image.

        The image must already be cropped and resized to the training
        resolution. MobileNetV2 normalization (ImageNet mean/std) is applied
        here before inference.

        Args:
            image_rgb (np.ndarray): Input image with shape ``(H, W, 3)`` and
                dtype ``uint8``.

        Returns:
            np.ndarray: Feature map with shape ``(H', W', C')`` and dtype
                ``float32``.

        Raises:
            RuntimeError: If ``load_backbone()`` has not been called.
        """
        if self._session is None:
            raise RuntimeError(
                "Backbone not loaded. Call load_backbone() before extract_features()."
            )

        tensor = self._preprocess(image_rgb)
        outputs = self._session.run(None, {self._input_name: tensor})
        # Output shape: (1, H', W', C') — drop batch dim for CalibrationService.
        feature_map: np.ndarray = outputs[0][0]          # (H', W', C')
        return feature_map

    # =========================================================================
    # Private helpers
    # =========================================================================

    @staticmethod
    def _preprocess(image_rgb: np.ndarray) -> np.ndarray:
        """
        Normalize image to MobileNetV2 ImageNet statistics and add batch dim.

        Args:
            image_rgb (np.ndarray): ``(H, W, 3)`` uint8 RGB image.

        Returns:
            np.ndarray: ``(1, H, W, 3)`` float32 NHWC tensor ready for ONNX.
        """
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = image_rgb.astype(np.float32) / 255.0
        img = (img - mean) / std        # HWC — backbone was exported as NHWC
        return img[np.newaxis, ...]     # HWC → NHWC
