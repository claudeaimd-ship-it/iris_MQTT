from abc import ABC, abstractmethod

import numpy as np


class IPaDiMFeatureExtractor(ABC):
    """
    Abstract interface for PaDiM backbone feature extraction.

    Used exclusively during calibration (sweep + fit). Inference uses the
    fitted Gaussian parameters stored in ``padim_*_params.npz``; it does not
    call this interface.

    Implementors must be able to load an ONNX backbone file and extract a
    spatial feature map from a single RGB image.
    """

    @abstractmethod
    def load_backbone(self, backbone_path: str) -> None:
        """
        Load the ONNX backbone for a specific MobileNetV2 block.

        Args:
            backbone_path (str): Absolute path to the ``.onnx`` file
                (e.g. ``data/models/backbones/teacher_mobilenetv2_backbone_b9.onnx``).

        Returns:
            None
        """
        pass

    @abstractmethod
    def extract_features(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Extract a spatial feature map from a single RGB image.

        Args:
            image_rgb (np.ndarray): Input image with shape ``(H, W, 3)`` and
                dtype ``uint8``. Must already be cropped and resized to the
                training resolution before calling this method.

        Returns:
            np.ndarray: Feature map with shape ``(H', W', C')`` and dtype
                ``float32``, where ``H'`` and ``W'`` are the spatial dimensions
                after the selected MobileNetV2 block, and ``C'`` is the number
                of feature channels at that block.
        """
        pass
