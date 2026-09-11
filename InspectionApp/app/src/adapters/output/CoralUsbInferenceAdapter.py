import json
import os
import numpy as np

from tflite_runtime.interpreter import Interpreter, load_delegate

from app.src.interfaces.IInferenceEngine import IInferenceEngine


class CoralUsbInferenceAdapter(IInferenceEngine):
    """
    Inference adapter for split-model execution on Coral USB Accelerator (EdgeTPU).

    Implements the teacher/student inference pattern:
    - Teacher (MobileNetV2 float32, CPU): extracts feature maps from the input image.
    - Student (autoencoder int8, EdgeTPU): reconstructs the feature maps.

    Both models are loaded from a single directory at startup. The adapter discovers
    all valid model pairs by scanning filenames that follow the convention::

        teacher_{view_name}_float32.tflite
        student_{view_name}_int8_edgetpu.tflite

    At prediction time, the correct pair is selected by the view_name built from
    ``section`` + ``_`` + ``channel``.

    Attributes:
        _models (dict[str, tuple[Interpreter, Interpreter]]): Loaded model pairs
            keyed by view_name (e.g. ``front_view_section_1_A``).
        _edgetpu_delegate: Cached EdgeTPU delegate (loaded once, reused for all
            student models to avoid repeated library initialization overhead).
    """

    _TEACHER_SUFFIX      = "_float32.tflite"
    _STUDENT_SUFFIX      = "_int8_edgetpu.tflite"
    _TEACHER_PREFIX      = "teacher_"
    _STUDENT_PREFIX      = "student_"
    _EVAL_CONFIG_FILENAME = "eval_config.json"

    _edgetpu_delegate = None  # Class-level cache — shared across all instances.

    def __init__(self):
        self._models: dict[str, tuple[Interpreter, Interpreter]] = {}
        self._eval_config: dict = {}

    # =========================================================================
    # IInferenceEngine interface
    # =========================================================================

    def load_models_from_directory(self, models_dir: str) -> None:
        """
        Scan ``models_dir`` for all teacher/student model pairs and load them.

        Each teacher file ``teacher_{view_name}_float32.tflite`` must have a
        matching student file ``student_{view_name}_int8_edgetpu.tflite`` in the
        same directory. Pairs with a missing counterpart are skipped with a warning.

        Args:
            models_dir (str): Directory containing the ``.tflite`` model files.

        Raises:
            FileNotFoundError: If ``models_dir`` does not exist.
            RuntimeError: If no valid model pairs are found.
        """
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f"Models directory not found: '{models_dir}'")

        delegate = self._get_edgetpu_delegate()

        # Discover all teacher files and look for a matching student file.
        teacher_files = [
            f for f in os.listdir(models_dir)
            if f.startswith(self._TEACHER_PREFIX) and f.endswith(self._TEACHER_SUFFIX)
        ]

        loaded = 0
        for teacher_filename in teacher_files:
            view_name = teacher_filename[
                len(self._TEACHER_PREFIX) : -len(self._TEACHER_SUFFIX)
            ]
            student_filename = f"{self._STUDENT_PREFIX}{view_name}{self._STUDENT_SUFFIX}"
            student_path = os.path.join(models_dir, student_filename)
            teacher_path = os.path.join(models_dir, teacher_filename)

            if not os.path.isfile(student_path):
                print(f"[WARN] No matching student model for '{teacher_filename}' — skipped.")
                continue

            interp_teacher = Interpreter(model_path=teacher_path)
            interp_teacher.allocate_tensors()

            interp_student = Interpreter(
                model_path=student_path,
                experimental_delegates=[delegate],
            )
            interp_student.allocate_tensors()

            self._models[view_name] = (interp_teacher, interp_student)
            loaded += 1
            print(f"[OK] Loaded model pair for view: '{view_name}'")

        if loaded == 0:
            print(
                f"[WARN] CoralUsbInferenceAdapter: no model pairs found in '{models_dir}'. "
                f"Expected teacher_{{view_name}}_float32.tflite and "
                f"student_{{view_name}}_int8_edgetpu.tflite. "
                f"Inference will fail until models are placed in this directory."
            )
            return

        print(f"[OK] {loaded} model pair(s) loaded from '{models_dir}'.")

        # Load model-level scoring configuration if present.
        eval_config_path = os.path.join(models_dir, self._EVAL_CONFIG_FILENAME)
        if os.path.isfile(eval_config_path):
            with open(eval_config_path, "r") as f:
                self._eval_config = json.load(f)
            print(f"[OK] eval_config.json loaded from '{models_dir}'.")
        else:
            print(f"[WARN] eval_config.json not found in '{models_dir}'. "
                  f"Default scoring parameters will be used.")
            self._eval_config = {}

    @property
    def eval_config(self) -> dict:
        """
        Model-level scoring parameters loaded from ``eval_config.json``.

        Expected keys (all optional; missing keys fall back to defaults in
        ``SequenceSettings``):

        - ``top_k_pixels`` (int): Number of top-error pixels for scoring.
        - ``border_crop_px`` (int): Pixels to crop from each border of the error map.
        - ``gaussian_sigma`` (float): Sigma for optional Gaussian blur on the error map
          (0.0 disables smoothing).

        Returns:
            dict: Parsed ``eval_config.json`` content, or ``{}`` if the file was
                not found.
        """
        return self._eval_config

    def predict(
        self,
        input_data: np.ndarray,
        section: str,
        channel: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run split inference: teacher extracts features, student reconstructs them.

        Args:
            input_data (np.ndarray): Preprocessed image, shape ``(1, H, W, 3)``,
                float32, MobileNetV2-normalized.
            section (str): View section, e.g. ``front_view_section_1``.
            channel (str): Camera channel, e.g. ``A``.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(features_original, features_reconstructed)``
                both float32 with shape ``(1, H', W', C')``.

        Raises:
            KeyError: If no model pair was loaded for this view_name.
        """
        view_name = f"{section}_{channel}"

        if view_name not in self._models:
            raise KeyError(
                f"No model loaded for view '{view_name}'. "
                f"Loaded views: {list(self._models.keys())}"
            )

        interp_teacher, interp_student = self._models[view_name]

        # === Step 1: Teacher (float32, CPU) — extract features ===
        teacher_input  = interp_teacher.get_input_details()[0]
        teacher_output = interp_teacher.get_output_details()[0]

        interp_teacher.set_tensor(teacher_input["index"], input_data.astype(np.float32))
        interp_teacher.invoke()
        features_original: np.ndarray = interp_teacher.get_tensor(teacher_output["index"]).copy()

        # === Step 2: Student (int8, EdgeTPU) — reconstruct features ===
        student_input  = interp_student.get_input_details()[0]
        student_output = interp_student.get_output_details()[0]

        # Quantize teacher float32 features to int8 for the student input.
        scale_in, zero_in = student_input["quantization"]
        features_quantized = np.clip(
            np.round(features_original / scale_in) + zero_in, -128, 127
        ).astype(np.int8)

        interp_student.set_tensor(student_input["index"], features_quantized)
        interp_student.invoke()
        reconstruction_int8: np.ndarray = interp_student.get_tensor(student_output["index"]).copy()

        # Dequantize student int8 output back to float32.
        scale_out, zero_out = student_output["quantization"]
        features_reconstructed = (reconstruction_int8.astype(np.float32) - zero_out) * scale_out

        return features_original, features_reconstructed

    # =========================================================================
    # Private helpers
    # =========================================================================

    @classmethod
    def _get_edgetpu_delegate(cls):
        """Load the EdgeTPU delegate once and cache it at class level."""
        if cls._edgetpu_delegate is None:
            print("[INFO] Loading EdgeTPU delegate...")
            cls._edgetpu_delegate = load_delegate("libedgetpu.so.1")
            print("[OK] EdgeTPU delegate loaded and cached.")
        return cls._edgetpu_delegate

