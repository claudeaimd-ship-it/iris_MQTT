import json
import os
import numpy as np
from typing import Any

from app.src.interfaces.IInferenceEngine import IInferenceEngine


def _new_interpreter(model_path: str) -> Any:
    """Instantiate a TFLite Interpreter using whichever backend is available.

    Defers the import to call time so that TensorFlow's lazy submodule loading
    has fully resolved ``tf.lite`` before the attribute is accessed.
    """
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf  # noqa: PLC0415
        return tf.lite.Interpreter(model_path=model_path)


class CpuInferenceAdapter(IInferenceEngine):
    """
    Inference adapter for split-model execution on CPU (no accelerator required).

    Functionally identical to ``CoralUsbInferenceAdapter`` but runs both teacher
    and student entirely on CPU using ``tflite_runtime``.  No EdgeTPU delegate
    is loaded, so this adapter works on any platform where ``tflite_runtime`` is
    installed (PC, Jetson, single-board computers without Coral hardware).

    Model naming convention (both models are float32)::

        teacher_{view_name}_float32.tflite
        student_{view_name}_float32.tflite

    The student model here is the full float32 autoencoder, NOT the int8
    EdgeTPU-compiled version.  The training pipeline must export both variants
    when targeting multi-platform deployment.

    Attributes:
        _models (dict[str, tuple[Any, Any]]): Loaded model pairs
            keyed by view_name (e.g. ``front_view_section_1_A``).
    """

    _TEACHER_SUFFIX       = "_float32.tflite"
    _STUDENT_SUFFIX       = "_float32.tflite"
    _TEACHER_PREFIX       = "teacher_"
    _STUDENT_PREFIX       = "student_"
    _EVAL_CONFIG_FILENAME = "eval_config.json"

    def __init__(self):
        self._models: dict[str, tuple[Any, Any]] = {}
        self._eval_config: dict = {}

    # =========================================================================
    # IInferenceEngine interface
    # =========================================================================

    def load_models_from_directory(self, models_dir: str) -> None:
        """
        Scan ``models_dir`` for all teacher/student float32 model pairs.

        Args:
            models_dir (str): Directory containing the ``.tflite`` model files.

        Raises:
            FileNotFoundError: If ``models_dir`` does not exist.
            RuntimeError: If no valid model pairs are found.
        """
        if not os.path.isdir(models_dir):
            raise FileNotFoundError(f"Models directory not found: '{models_dir}'")

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
            student_path     = os.path.join(models_dir, student_filename)
            teacher_path     = os.path.join(models_dir, teacher_filename)

            if not os.path.isfile(student_path):
                print(f"[WARN] No matching CPU student model for '{teacher_filename}' — skipped.")
                continue

            # Skip EdgeTPU-compiled student files that also end in _float32.tflite by accident.
            # (Unlikely, but guard against teacher file matching its own student slot.)
            if teacher_path == student_path:
                continue

            interp_teacher = _new_interpreter(teacher_path)
            interp_teacher.allocate_tensors()

            interp_student = _new_interpreter(student_path)
            interp_student.allocate_tensors()

            self._models[view_name] = (interp_teacher, interp_student)
            loaded += 1
            print(f"[OK] CpuInferenceAdapter: loaded model pair for view '{view_name}'")

        if loaded == 0:
            print(
                f"[WARN] CpuInferenceAdapter: no model pairs found in '{models_dir}'. "
                f"Expected teacher_{{view_name}}_float32.tflite and "
                f"student_{{view_name}}_float32.tflite. "
                f"Inference will fail until models are placed in this directory."
            )
            return

        print(f"[OK] CpuInferenceAdapter: {loaded} model pair(s) loaded from '{models_dir}'.")

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
        Run split inference entirely on CPU (float32 → float32, no quantization).

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
                f"No CPU model loaded for view '{view_name}'. "
                f"Loaded views: {list(self._models.keys())}"
            )

        interp_teacher, interp_student = self._models[view_name]

        # === Step 1: Teacher (float32, CPU) — extract features ===
        teacher_input  = interp_teacher.get_input_details()[0]
        teacher_output = interp_teacher.get_output_details()[0]

        interp_teacher.set_tensor(teacher_input["index"], input_data.astype(np.float32))
        interp_teacher.invoke()
        features_original: np.ndarray = interp_teacher.get_tensor(teacher_output["index"]).copy()

        # === Step 2: Student (float32, CPU) — reconstruct features ===
        student_input  = interp_student.get_input_details()[0]
        student_output = interp_student.get_output_details()[0]

        interp_student.set_tensor(student_input["index"], features_original.astype(np.float32))
        interp_student.invoke()
        features_reconstructed: np.ndarray = interp_student.get_tensor(student_output["index"]).copy()

        return features_original, features_reconstructed
