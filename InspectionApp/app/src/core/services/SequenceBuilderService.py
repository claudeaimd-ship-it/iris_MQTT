import copy
import json
import os
from datetime import datetime


class SequenceBuilderService:
    """
    Core service for building, validating, and persisting inspection sequences.

    Receives a wizard state dict and produces a validated sequence JSON that
    can be consumed by AppFactory. This service knows the schema of the
    sequence JSON but has no dependency on adapters or hardware.

    All mutating methods return a new copy of the draft — the caller is
    responsible for persisting the draft to disk after each wizard step via
    ``save_draft()``.

    Attributes:
        REQUIRED_PIPELINE_TOOL (str): Tool that must appear in every camera
            pipeline. Without it the training pipeline cannot determine the
            model input resolution.
        INFERENCE_STEP_MIN (int): Minimum step_number for inference steps.
        NOK_STEP_MAX (int): Maximum step_number (exclusive upper bound) for
            NOK-dispatch steps.
    """

    REQUIRED_PIPELINE_TOOL: str = "resize_to_training_resolution"
    INFERENCE_STEP_MIN: int = 1001
    NOK_STEP_MAX: int = 0

    # =========================================================================
    # Draft construction
    # =========================================================================

    def create_draft(self, hardware_config: dict) -> dict:
        """
        Build the initial draft from the hardware wizard config.

        Args:
            hardware_config (dict): Must contain keys:
                - ``part_model`` (str)
                - ``device`` (str)
                - ``camera_type`` (str)
                - ``camera_model`` (str)
                - ``number_of_cameras`` (int)
                - ``camera_port`` (list[str])
                - ``camera_capture_resolution`` (list[int, int])
                - ``camera_preview_parameters`` (dict)
                - ``inference_device`` (str)
                - ``io_module`` (str)
                - ``trigger_input_pin`` (int)
                - ``spotlight_gpio_pins`` (list[int])
                - ``gpio_configuration`` (list[dict])
                - ``paths`` (dict)

        Returns:
            dict: Initial draft with empty pipeline and steps list.
        """
        return {
            "project_name": hardware_config.get("part_model", "New Project"),
            "part_model": hardware_config.get("part_model", ""),
            "description": "",
            "paths": hardware_config.get("paths", {
                "images_path": "./data/images/",
                "target_train_images": 1500,
                "inference_images_path": "./data/images/inference/",
                "traceability_inference_path": "./data/traceability/inference/",
                "model_path": "./data/models/",
            }),
            "hardware": {
                "device_type": hardware_config.get("device_type", "PC"),
                "camera_type": hardware_config.get("camera_type", "USB"),
                "camera_model": hardware_config.get("camera_model", ""),
                "camera_index": hardware_config.get("camera_index", 0),
                "number_of_cameras": hardware_config.get("number_of_cameras", 1),
                "camera_port": hardware_config.get("camera_port", ["A"]),
                "camera_capture_resolution": hardware_config.get("camera_capture_resolution", [4608, 2592]),
                "camera_preview_parameters": hardware_config.get("camera_preview_parameters", {
                    "resolution": [320, 180],
                    "exposure_time": 0,
                    "lens_position": 0,
                }),
                "inference_device": hardware_config.get("inference_device", "Coral USB Accelerator"),
                "io_module": hardware_config.get("io_module", ""),
                "trigger_input_pin": hardware_config.get("trigger_input_pin", 0),
                "spotlight_gpio_pins": hardware_config.get("spotlight_gpio_pins", []),
                "gpio_configuration": hardware_config.get("gpio_configuration", []),
            },
            "preprocessing_image_parameters": [],
            "scoring": {
                "top_k_pixels": 15,
            },
            "steps": [],
        }

    def update_pipeline(
        self,
        draft: dict,
        camera_port: str,
        view: str,
        section: str,
        pipeline: list[dict],
        inference_type: str = "standard",
    ) -> dict:
        """
        Set or replace the preprocessing pipeline for one camera/section.

        Args:
            draft (dict): Current sequence draft.
            camera_port (str): Camera channel identifier (e.g. ``"A"``).
            view (str): View name prefix (e.g. ``"front_view"``).
            section (str): Section number as string (e.g. ``"1"``).
            pipeline (list[dict]): Ordered list of tool dicts. Each dict must
                have a ``"tool"`` key and an optional ``"parameters"`` key.
            inference_type (str): ``"standard"`` (default) or
                ``"presence_detection_absence_calibrated"`` — inverts the
                OK/NOK decision for this view (see ``InspectionService``).

        Returns:
            dict: New draft with the pipeline replaced.
        """
        draft = copy.deepcopy(draft)
        params = draft.setdefault("preprocessing_image_parameters", [])

        # Replace existing entry for this camera/section or append a new one.
        for entry in params:
            if entry.get("camera_port") == camera_port and entry.get("section") == section:
                entry["pipeline"] = pipeline
                entry["view"] = view
                entry["inference_type"] = inference_type
                return draft

        params.append({
            "camera_port": camera_port,
            "view": view,
            "section": section,
            "pipeline": pipeline,
            "inference_type": inference_type,
        })
        return draft

    def add_step(self, draft: dict, step: dict) -> dict:
        """
        Append a step to the sequence steps list.

        The step is inserted in ascending order of ``step_number``. Steps with
        the same ``step_number`` replace the existing one.

        Args:
            draft (dict): Current sequence draft.
            step (dict): Step dict with at least a ``"step_number"`` key.

        Returns:
            dict: New draft with the step added.
        """
        draft = copy.deepcopy(draft)
        steps = draft.setdefault("steps", [])
        step_number = step["step_number"]

        # Replace if a step with the same number already exists.
        for i, existing in enumerate(steps):
            if existing["step_number"] == step_number:
                steps[i] = step
                return draft

        steps.append(step)
        steps.sort(key=lambda s: s["step_number"])
        return draft

    def remove_step(self, draft: dict, step_number: int) -> dict:
        """
        Remove a step by its step_number.

        Args:
            draft (dict): Current sequence draft.
            step_number (int): The step_number to remove.

        Returns:
            dict: New draft without the specified step.
        """
        draft = copy.deepcopy(draft)
        draft["steps"] = [s for s in draft.get("steps", []) if s["step_number"] != step_number]
        return draft

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self, draft: dict) -> list[str]:
        """
        Validate the draft and return a list of human-readable error strings.

        An empty list means the draft is valid and can be saved as a final
        sequence. All errors are collected before returning so the user can
        fix them all at once.

        Args:
            draft (dict): Sequence draft to validate.

        Returns:
            list[str]: Validation error messages. Empty if valid.
        """
        errors: list[str] = []

        # ── Basic fields ──────────────────────────────────────────────────────
        if not draft.get("part_model", "").strip():
            errors.append("Part model name is required.")

        # ── Hardware ─────────────────────────────────────────────────────────
        hw = draft.get("hardware", {})
        n_cameras = hw.get("number_of_cameras", 0)
        ports = hw.get("camera_port", [])
        if len(ports) != n_cameras:
            errors.append(
                f"number_of_cameras is {n_cameras} but {len(ports)} camera ports are defined. "
                "They must match exactly."
            )

        # GPIO validations only apply when GPIO is configured (not PC)
        device_type = hw.get("device_type", "")
        gpio_config = hw.get("gpio_configuration", [])
        requires_gpio = device_type.lower() != "pc" or len(gpio_config) > 0

        if requires_gpio:
            trigger_pin = hw.get("trigger_input_pin", 0)
            if trigger_pin == 0:
                errors.append("trigger_input_pin must be set (non-zero).")

            gpio_pins = {g["pin_number"] for g in gpio_config}
            if trigger_pin not in gpio_pins:
                errors.append(
                    f"trigger_input_pin {trigger_pin} is not in gpio_configuration."
                )

            # Verify trigger pin is declared as input.
            gpio_type_map = {
                g["pin_number"]: g.get("type")
                for g in gpio_config
            }
            if gpio_type_map.get(trigger_pin) != "input":
                errors.append(
                    f"trigger_input_pin {trigger_pin} must be of type 'input' in gpio_configuration."
                )

            has_output_pin = any(
                g.get("type") == "output" for g in gpio_config
            )
            if not has_output_pin:
                errors.append("gpio_configuration must have at least one output pin.")

        # ── Preprocessing pipelines ───────────────────────────────────────────
        pipelines = draft.get("preprocessing_image_parameters", [])
        if not pipelines:
            errors.append(
                "No preprocessing pipelines defined. Add at least one camera view pipeline "
                "in the builder before saving."
            )

        for entry in pipelines:
            cam = entry.get("camera_port", "?")
            sec = entry.get("section", "?")
            pipeline = entry.get("pipeline", [])
            tool_names = [t.get("tool") for t in pipeline]

            if self.REQUIRED_PIPELINE_TOOL not in tool_names:
                errors.append(
                    f"Pipeline for camera {cam}, section {sec} is missing required tool "
                    f"'{self.REQUIRED_PIPELINE_TOOL}' (sets model input resolution)."
                )
            else:
                # Verify width and height are set.
                for tool in pipeline:
                    if tool.get("tool") == self.REQUIRED_PIPELINE_TOOL:
                        params = tool.get("parameters", {})
                        if not params.get("width") or not params.get("height"):
                            errors.append(
                                f"'{self.REQUIRED_PIPELINE_TOOL}' in camera {cam}, section {sec} "
                                "must specify both 'width' and 'height'."
                            )

        # ── Steps ─────────────────────────────────────────────────────────────
        steps = draft.get("steps", [])
        step_numbers = [s["step_number"] for s in steps]

        if not any(n >= self.INFERENCE_STEP_MIN for n in step_numbers):
            errors.append(
                f"No inference step found (step_number >= {self.INFERENCE_STEP_MIN}). "
                "Add at least one inference step. See inference_sequence_example_editable.jsonc "
                "for the required structure."
            )

        if not any(n < self.NOK_STEP_MAX for n in step_numbers):
            errors.append(
                f"No NOK-dispatch step found (step_number < {self.NOK_STEP_MAX}). "
                "Add at least one step with a negative step_number to dispatch rejected parts. "
                "See inference_sequence_example_editable.jsonc for the required structure."
            )

        # Verify GPIO pins used in steps exist in gpio_configuration (only if GPIO is configured).
        if requires_gpio:
            gpio_pins = {g["pin_number"] for g in gpio_config}
            for step in steps:
                for gpio_action in step.get("gpio_action", []):
                    pin = gpio_action.get("pin_number")
                    if pin and pin not in gpio_pins:
                        errors.append(
                            f"Step {step['step_number']}: pin {pin} is not declared in "
                            "gpio_configuration."
                        )

        # Verify camera ports used in steps are declared.
        declared_ports = set(hw.get("camera_port", []))
        for step in steps:
            for cam_action in step.get("camera_action", []):
                port = cam_action.get("camera_port")
                if port and port not in declared_ports:
                    errors.append(
                        f"Step {step['step_number']}: camera_port '{port}' is not in the "
                        "declared camera_port list."
                    )

        # At most one detect_piece_action step per sequence.
        detect_piece_steps = [s for s in steps if "detect_piece_action" in s]
        if len(detect_piece_steps) > 1:
            errors.append("At most one detect_piece_action step is allowed per sequence.")
        for step in detect_piece_steps:
            piece_action = step["detect_piece_action"]
            port = piece_action.get("camera_port")
            if port and port not in declared_ports:
                errors.append(
                    f"Step {step['step_number']}: detect_piece_action camera_port '{port}' "
                    "is not in the declared camera_port list."
                )
            roi = piece_action.get("roi", {})
            if not roi or roi.get("w", 0) <= 0 or roi.get("h", 0) <= 0:
                errors.append(
                    f"Step {step['step_number']}: detect_piece_action is missing a valid Detection ROI "
                    "(w and h must be > 0). Draw the orange Detection ROI on the canvas."
                )

        # At most one wait_for_piece_action step per sequence.
        wait_for_piece_steps = [s for s in steps if "wait_for_piece_action" in s]
        if len(wait_for_piece_steps) > 1:
            errors.append("At most one wait_for_piece_action step is allowed per sequence.")
        for step in wait_for_piece_steps:
            wfp_action = step["wait_for_piece_action"]
            port = wfp_action.get("camera_port")
            if port and port not in declared_ports:
                errors.append(
                    f"Step {step['step_number']}: wait_for_piece_action camera_port '{port}' "
                    "is not in the declared camera_port list."
                )
            roi = wfp_action.get("roi", {})
            if not roi or roi.get("w", 0) <= 0 or roi.get("h", 0) <= 0:
                errors.append(
                    f"Step {step['step_number']}: wait_for_piece_action is missing a valid Detection ROI "
                    "(w and h must be > 0). Draw the orange Detection ROI on the canvas."
                )
            if wfp_action.get("pixel_diff_threshold", 0) <= 0:
                errors.append(
                    f"Step {step['step_number']}: wait_for_piece_action pixel_diff_threshold must be > 0."
                )

        return errors

    # =========================================================================
    # Persistence
    # =========================================================================

    def save_draft(self, draft: dict, path: str) -> None:
        """
        Persist the current draft to disk (overwrites).

        Used after each wizard step so the draft survives page reloads.

        Args:
            draft (dict): Sequence draft to persist.
            path (str): File path (e.g. ``"config/sequence_draft.json"``).

        Returns:
            None
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(draft, f, indent=4, ensure_ascii=False)

    def load_draft(self, path: str) -> dict:
        """
        Load a draft from disk.

        Args:
            path (str): File path to read.

        Returns:
            dict: Draft dict. Returns an empty dict if the file does not exist.
        """
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_sequence(self, draft: dict, sequences_dir: str) -> str:
        """
        Validate the draft and save it as a final numbered sequence file.

        The file is named ``sequence_NNN.json`` where NNN is the next
        available 3-digit number in ``sequences_dir``.

        Args:
            draft (dict): Sequence draft to save.
            sequences_dir (str): Directory where sequence files live
                (e.g. ``"config/"``).

        Returns:
            str: Absolute path to the saved sequence file.

        Raises:
            ValueError: If the draft fails validation. The error message
                contains all validation errors separated by newlines.
        """
        errors = self.validate(draft)
        if errors:
            raise ValueError(
                "Sequence validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        os.makedirs(sequences_dir, exist_ok=True)
        next_n = self._next_sequence_number(sequences_dir)
        filename = f"sequence_{next_n:03d}.json"
        path = os.path.join(sequences_dir, filename)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(draft, f, indent=4, ensure_ascii=False)

        print(f"[OK] SequenceBuilderService: sequence saved to '{path}'.")
        return os.path.abspath(path)

    # =========================================================================
    # Helpers
    # =========================================================================

    def list_sequences(self, sequences_dir: str) -> list[dict]:
        """
        Return metadata for all saved sequence files in ``sequences_dir``.
        Excludes ``sequence_draft.json`` (draft is not a final saved sequence).

        Args:
            sequences_dir (str): Directory to scan (e.g. ``"config/"``).

        Returns:
            list[dict]: Each entry has ``path``, ``filename``, and
                ``part_model`` keys, sorted by filename.
        """
        result: list[dict] = []
        if not os.path.isdir(sequences_dir):
            return result

        for filename in sorted(os.listdir(sequences_dir)):
            if filename == "sequence_draft.json":
                continue  # Skip draft file
            if not (filename.startswith("sequence_") and filename.endswith(".json")):
                continue
            full_path = os.path.join(sequences_dir, filename)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                result.append({
                    "path": full_path,
                    "filename": filename,
                    "part_model": data.get("part_model", filename),
                    "modified": datetime.fromtimestamp(
                        os.path.getmtime(full_path)
                    ).strftime("%Y-%m-%d %H:%M"),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return result

    def find_sequence_by_part_model(
        self, part_model: str, sequences_dir: str
    ) -> dict | None:
        """
        Return the first sequence file whose ``part_model`` matches the given
        value, or ``None`` if no match is found.

        Args:
            part_model (str): Part model name to search for.
            sequences_dir (str): Directory to scan (e.g. ``"config/"``).

        Returns:
            dict | None: Metadata dict (``path``, ``filename``, ``part_model``)
                if found, else ``None``.
        """
        for seq in self.list_sequences(sequences_dir):
            if seq.get("part_model", "").strip() == part_model.strip():
                return seq
        return None

    def _next_sequence_number(self, sequences_dir: str) -> int:
        """Return the next available sequence file number, starting at 1."""
        existing: list[int] = []
        for filename in os.listdir(sequences_dir):
            if filename.startswith("sequence_") and filename.endswith(".json"):
                stem = filename[len("sequence_"):-len(".json")]
                if stem.isdigit():
                    existing.append(int(stem))
        return max(existing, default=0) + 1
