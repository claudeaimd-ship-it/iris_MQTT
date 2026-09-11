# Iris — Vision Inspection System
# Author : Alan Rojas — Team MDA & DA, Johnson Electric, Zacatecas, México
# Project: InspectionApp (Raspberry Pi 4B · PaDiM · Hexagonal Architecture)

import copy
import json
import logging
import os
import socket
import time
import threading
import traceback

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, send_file, url_for
from flask_cors import CORS

# Suppress werkzeug's per-request access log so the 3-second status poll from
# the browser topbar does not flood TimestampedFileLogger. Errors (500, etc.)
# are still printed because Flask logs them separately via app.logger.
logging.getLogger("werkzeug").setLevel(logging.ERROR)

from app.src.AppFactory import AppFactory
from app.src.core.services.SequenceBuilderService import SequenceBuilderService
from app.src.core.settings.SequenceSettings import SequenceSettings
from app.src.core.utils.TimestampedFileLogger import TimestampedFileLogger
from app.src.core.utils.ViewConfigBuilder import build_view_configs, build_view_names
from iris.IrisState import IrisState


_DEFAULT_VALUES_PATH = "config/default_values.json"
_SEQUENCES_DIR       = "config/"
_DRAFT_PATH          = "config/sequence_draft.json"
_SESSION_STATE_PATH  = "config/session_state.json"
_VERSION_PATH        = "app/VERSION"
_STREAM_FPS          = 5
_JPEG_QUALITY        = 65
_APP_START_TIME      = time.time()  # For uptime calculation in /api/info


def _read_iris_version() -> str:
    """Reads the semver string from app/VERSION (source of truth for
    /api/info's "iris_version"). Falls back to "unknown" instead of crashing
    the app if the file is missing, since this value is informational only."""
    try:
        with open(_VERSION_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "unknown"


_IRIS_VERSION = _read_iris_version()

_connection_lock     = threading.Lock()
_stream_connections  = 0


def create_iris_app() -> Flask:
    """
    Build and configure the Iris Flask application.

    All route handlers close over ``state`` and ``builder_service``.
    The app is intentionally NOT a Flask blueprint so the single-page
    structure stays obvious and easy to follow.

    Returns:
        Flask: Configured Flask application ready to call ``run()``.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.secret_key = os.urandom(24)

    # Enable CORS for Abigail (fleet management server) to consume Iris API.
    # In production, restrict origins to the Abigail server IP.
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    state           = IrisState(draft_path=_DRAFT_PATH)
    builder_service = SequenceBuilderService()

    # Camera keepalive: reads one preview frame every 5 min while all loops are
    # idle to prevent the libcamera ISP pipeline from stalling after extended
    # periods of inactivity (observed after 3-4 days without use).
    threading.Thread(
        target=_camera_keepalive,
        args=(state,),
        daemon=True,
        name="CameraKeepalive",
    ).start()

    # In-process cycle watchdog: force-restarts the process if an inspection
    # cycle hangs past cycle_watchdog_timeout_s (config/default_values.json).
    # See _cycle_watchdog() docstring for why this only counts time since the
    # trigger was received, never idle time waiting for the next part.
    threading.Thread(
        target=_cycle_watchdog,
        args=(state,),
        daemon=True,
        name="CycleWatchdog",
    ).start()

    # Load read-only catalog data once at startup.
    with open("config/camera_catalog.json",  "r", encoding="utf-8") as _f:
        _camera_catalog = json.load(_f)
    with open("config/io_module_catalog.json", "r", encoding="utf-8") as _f:
        _io_catalog = json.load(_f)
    with open("config/device_catalog.json",   "r", encoding="utf-8") as _f:
        _device_catalog = json.load(_f)

    # If at least one sequence exists, pre-load it.
    _existing = builder_service.list_sequences(_SEQUENCES_DIR)
    if _existing:
        try:
            _load_sequence(state, _existing[-1]["path"], warmup=False)
        except Exception as _e:
            print(f"[WARN] IrisServer: could not pre-load sequence on startup: {_e}")

    # =========================================================================
    # Pages
    # =========================================================================

    @app.route("/")
    def home():
        sequences = builder_service.list_sequences(_SEQUENCES_DIR)
        # If at least one sequence has already been saved, skip the picker
        # and go straight to inspection so operators don't have to click
        # through Home on every visit. The top-nav "Home" link passes
        # ?manage=1 to bypass this and always reach the picker below — it's
        # the only entry point to "Edit sequence"/"Clone as new part", so it
        # must stay reachable even when a sequence is already loaded (plain
        # visits to "/", e.g. the kiosk browser's start URL, still redirect).
        if sequences and not request.args.get("manage"):
            return redirect(url_for("inspection"))

        # Check for an in-progress builder draft so the home page can offer
        # a "Resume Draft" shortcut without going through the setup wizard.
        draft_info = None
        if os.path.isfile(_DRAFT_PATH):
            _d = builder_service.load_draft(_DRAFT_PATH)
            if _d.get("part_model"):
                _ep = _d.get("_edit_path")
                draft_info = {
                    "part_model":   _d["part_model"],
                    "edit_path":    _ep,
                    "edit_filename": os.path.basename(_ep) if _ep else None,
                }
        return render_template(
            "home.html",
            sequences=sequences,
            current_sequence=state.current_sequence_path,
            draft_info=draft_info,
        )

    @app.route("/setup")
    def setup():
        draft = builder_service.load_draft(_DRAFT_PATH)
        return render_template(
            "setup.html",
            camera_catalog=_camera_catalog,
            io_catalog=_io_catalog,
            device_catalog=_device_catalog,
            draft=draft,
        )

    @app.route("/builder")
    def builder():
        with state.lock:
            has_seq = state.has_sequence
            draft   = state.draft
        if not has_seq:
            return redirect(url_for("setup"))
        if draft is None:
            draft = builder_service.load_draft(_DRAFT_PATH)
        return render_template("builder.html", draft=draft, io_catalog=_io_catalog)

    @app.route("/inspection")
    def inspection():
        sequences = builder_service.list_sequences(_SEQUENCES_DIR)
        if not sequences:
            return redirect(url_for("home"))
        with state.lock:
            has_seq  = state.has_sequence
            seq_path = state.current_sequence_path
        if not has_seq:
            try:
                _load_sequence(state, sequences[-1]["path"])
                seq_path = sequences[-1]["path"]
            except Exception as _e:
                return render_template("home.html", sequences=sequences, error=str(_e))
        return render_template(
            "inspection.html",
            sequences=sequences,
            current_sequence=os.path.basename(seq_path) if seq_path else None,
        )

    # =========================================================================
    # Setup API
    # =========================================================================

    @app.route("/api/info")
    def api_info():
        """
        Return static metadata about this Iris instance for Abigail discovery
        and monitoring. Useful for fleet-level dashboards.
        """
        hostname = socket.gethostname()
        with state.lock:
            factory = state.factory
            seq_path = state.current_sequence_path
        
        # Count available sequences
        sequences = builder_service.list_sequences(_SEQUENCES_DIR)
        
        # Determine device type from the loaded sequence or default config
        device_type = "RaspberryPi"  # default
        if factory and seq_path:
            try:
                with open(seq_path, "r", encoding="utf-8") as f:
                    seq_data = json.load(f)
                    device_type = seq_data.get("hardware", {}).get("device_type", "RaspberryPi")
            except Exception:
                pass
        
        # Uptime: time since process start (approximation)
        # For true system uptime on Linux, read /proc/uptime
        uptime_seconds = int(time.time() - _APP_START_TIME)
        
        return jsonify({
            "hostname": hostname,
            "iris_version": _IRIS_VERSION,
            "device_type": device_type,
            "uptime_seconds": uptime_seconds,
            "available_sequences": len(sequences),
        })

    @app.route("/api/validate_part_model", methods=["POST"])
    def api_validate_part_model():
        """
        Validate if a part_model name is already in use by a saved sequence.
        Excludes the draft file (sequence_draft.json) from the search.
        Returns {"exists": true/false}.
        """
        data = request.get_json(force=True)
        part_model = (data.get("part_model") or "").strip()
        if not part_model:
            return jsonify({"exists": False})
        
        duplicate = builder_service.find_sequence_by_part_model(part_model, _SEQUENCES_DIR)
        return jsonify({"exists": duplicate is not None})

    @app.route("/api/setup/finalize", methods=["POST"])
    def api_setup_finalize():
        """
        Receive the wizard hardware config, create the initial draft, initialize
        hardware, and redirect to /builder. Paths are generated automatically
        based on part_model.
        """
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty request body."}), 400

        # Reject duplicate part_model names to prevent config/data overlap.
        part_model = (data.get("part_model") or "").strip()
        if not part_model:
            return jsonify({"error": "Product model name is required."}), 400

        duplicate = builder_service.find_sequence_by_part_model(part_model, _SEQUENCES_DIR)
        if duplicate:
            return jsonify({
                "error": (
                    f"A sequence with model name '{part_model}' already exists "
                    f"({duplicate['filename']}). Use a different name or load the existing sequence."
                )
            }), 409

        # Generate paths automatically based on part_model
        data["paths"] = _generate_paths_for_part_model(part_model)

        draft = builder_service.create_draft(data)

        # Create all storage directories declared in the wizard before
        # initialising hardware so AppFactory can find the models_dir.
        paths = draft.get("paths", {})
        for key in ("images_path", "inference_images_path",
                    "traceability_inference_path", "model_path"):
            dir_path = paths.get(key, "")
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

        builder_service.save_draft(draft, _DRAFT_PATH)

        with state.lock:
            if state.is_busy:
                return jsonify({"error": "Stop the loop before loading a new sequence."}), 409
            if state.factory is not None:
                state.factory.shutdown()
            try:
                state.factory = AppFactory(_DEFAULT_VALUES_PATH, _DRAFT_PATH)
                state.factory.create_inference_controller()
                state.factory.initialize_hardware()
                state.current_sequence_path = _DRAFT_PATH
                state.draft = draft
                _sync_capture_resolution(state)
            except Exception as exc:
                state.factory = None
                return jsonify({"error": str(exc)}), 500

        return jsonify({"redirect": url_for("builder")})

    # =========================================================================
    # Builder API
    # =========================================================================

    @app.route("/api/builder/toggle_gpio", methods=["POST"])
    def api_builder_toggle_gpio():
        """
        Toggle a specific GPIO pin on/off (used for manual spotlight control in the builder).
        Body: {pin: int, state: "on" | "off"}
        """
        data = request.get_json(force=True)
        pin = data.get("pin")
        action = data.get("state")

        if pin is None or action not in ("on", "off"):
            return jsonify({"error": "Missing or invalid 'pin' or 'state'."}), 400
        
        with state.lock:
            factory = state.factory
            if not factory or not factory._gpio:
                return jsonify({"error": "Hardware not initialized or GPIO not supported."}), 400
            
            try:
                if action == "on":
                    factory._gpio.turn_on(pin)
                else:
                    factory._gpio.turn_off(pin)
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500
            
        return jsonify({"ok": True, "pin": pin, "state": action})

    @app.route("/api/builder/capture_preview", methods=["POST"])
    def api_builder_capture_preview():
        """
        Capture one high-res frame per camera channel and store as JPEG.
        Extracts exposure/lens parameters from the active section's pipeline if available, otherwise uses the last captured settings or defaults.
        Returns a list of captured channels and any per-channel errors.
        """
        data = request.get_json(silent=True) or {}
        section_id = data.get("section")

        with state.lock:
            if not state.has_sequence:
                return jsonify({"error": "No sequence loaded."}), 400
            factory = state.factory
            current_draft = state.draft or {}

        channels = factory.get_camera_ports()
        errors: dict[str, str] = {}

        # Pre-load pipelines of active section to apply the correct exposure/lens settings for preview capture.
        pipelines_by_port = {}
        if section_id is not None:
            params_list = current_draft.get("preprocessing_image_parameters")
            if not params_list:
                params_list = factory._sequence.get("preprocessing_image_parameters", [])
                
            for entry in params_list:
                if str(entry.get("section")) == str(section_id):
                    pipelines_by_port[entry.get("camera_port")] = entry.get("pipeline", [])

        for channel in channels:
            try:
                pipe = pipelines_by_port.get(channel) or []

                exposure = None
                lens_position = None
                for tool in pipe:
                    if tool.get("tool") == "set_time_exposure":
                        exposure = tool.get("parameters", {}).get("exposure_time")
                    elif tool.get("tool") == "set_lens_position":
                        lens_position = tool.get("parameters", {}).get("lens_position")

                frame = factory.capture_preview_frame(channel, exposure_time=exposure, lens_position=lens_position)
                _, jpeg = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                state.store_frame(channel, jpeg.tobytes())
            except Exception as exc:
                print(f"[ERROR] Failed to capture preview for channel '{channel}': {exc}")
                traceback.print_exc()
                errors[channel] = str(exc)

        return jsonify({
            "captured": channels,
            "errors": errors,
            "hardware_wedged_message": factory.get_hardware_wedged_message(),
        })

    @app.route("/api/builder/frame/<channel>")
    def api_builder_frame(channel: str):
        """Return the last JPEG captured for a camera channel."""
        jpeg = state.get_frame(channel)
        if jpeg is None:
            return Response(status=204)
        return Response(jpeg, mimetype="image/jpeg")

    @app.route("/api/builder/update_pipeline", methods=["POST"])
    def api_builder_update_pipeline():
        """
        Set or replace the preprocessing pipeline for one camera / section.
        Body: {camera_port, view, section, pipeline: [{tool, parameters?}, ...], inference_type?}
        """
        data = request.get_json(force=True)
        required = {"camera_port", "view", "section", "pipeline"}
        if not required.issubset(data.keys()):
            return jsonify({"error": f"Missing fields: {required - data.keys()}"}), 400

        with state.lock:
            draft = state.draft or builder_service.load_draft(_DRAFT_PATH)
            draft = builder_service.update_pipeline(
                draft,
                camera_port=data["camera_port"],
                view=data["view"],
                section=str(data["section"]),
                pipeline=data["pipeline"],
                inference_type=data.get("inference_type", "standard"),
            )
            state.draft = draft
            builder_service.save_draft(draft, _DRAFT_PATH)

        return jsonify({"ok": True})
    
    @app.route("/api/builder/update_hardware", methods=["POST"])
    def api_builder_update_hardware():
        """
        Update hardware settings from the builder and persist to the draft.
        Body: {spotlights: [int], trigger_pin: int, capture_res: [width, height],
               gpio_configuration: [dict]}
        """
        data = request.get_json(force=True)
        with state.lock:
            draft = state.draft or builder_service.load_draft(_DRAFT_PATH)
            if "hardware" not in draft:
                draft["hardware"] = {}

            if "spotlights" in data:
                draft["hardware"]["spotlight_gpio_pins"] = data["spotlights"]
            if "trigger_pin" in data:
                draft["hardware"]["trigger_input_pin"] = data["trigger_pin"]
            if "capture_res" in data:
                draft["hardware"]["camera_capture_resolution"] = data["capture_res"]
            if "gpio_configuration" in data:
                draft["hardware"]["gpio_configuration"] = data["gpio_configuration"]

            state.draft = draft
            builder_service.save_draft(draft, _DRAFT_PATH)

        return jsonify({"ok": True, "hardware": draft["hardware"]})

    @app.route("/api/builder/add_step", methods=["POST"])
    def api_builder_add_step():
        """
        Add or replace a step in the draft.
        Body: {step: {step_number, description, ...}}
        """
        data = request.get_json(force=True)
        if "step" not in data:
            return jsonify({"error": "Missing 'step' field."}), 400

        with state.lock:
            draft = state.draft or builder_service.load_draft(_DRAFT_PATH)
            draft = builder_service.add_step(draft, data["step"])
            state.draft = draft
            builder_service.save_draft(draft, _DRAFT_PATH)

        return jsonify({"ok": True, "steps": draft["steps"]})

    @app.route("/api/builder/remove_step", methods=["POST"])
    def api_builder_remove_step():
        """
        Remove a step by step_number.
        Body: {step_number: int}
        """
        data = request.get_json(force=True)
        if "step_number" not in data:
            return jsonify({"error": "Missing 'step_number'."}), 400

        with state.lock:
            draft = state.draft or builder_service.load_draft(_DRAFT_PATH)
            draft = builder_service.remove_step(draft, data["step_number"])
            state.draft = draft
            builder_service.save_draft(draft, _DRAFT_PATH)

        return jsonify({"ok": True, "steps": draft["steps"]})

    @app.route("/api/builder/validate", methods=["POST"])
    def api_builder_validate():
        """Validate the current draft and return errors (empty list = valid)."""
        with state.lock:
            draft = state.draft or builder_service.load_draft(_DRAFT_PATH)
        errors = builder_service.validate(draft)
        return jsonify({"valid": len(errors) == 0, "errors": errors})

    @app.route("/api/builder/save_sequence", methods=["POST"])
    def api_builder_save_sequence():
        """
        Validate the draft and persist it.

        In edit mode (the draft carries ``_edit_path``) the original sequence
        file is overwritten so no new numbered file is created. Otherwise a
        new auto-numbered ``sequence_NNN.json`` is written to ``_SEQUENCES_DIR``.
        On success reloads the factory with the saved sequence.
        """
        with state.lock:
            if state.is_busy:
                return jsonify({"error": "Stop the loop before saving a new sequence."}), 409
            draft = state.draft or builder_service.load_draft(_DRAFT_PATH)

        edit_path = draft.get("_edit_path")
        if edit_path:
            # Edit mode: strip the internal metadata field and overwrite the
            # original sequence file instead of creating a new numbered one.
            save_draft = {k: v for k, v in draft.items() if k != "_edit_path"}
            errors = builder_service.validate(save_draft)
            if errors:
                return jsonify({"error": "Validation failed:\n" + "\n".join(errors)}), 400
            try:
                with open(edit_path, "w", encoding="utf-8") as _f:
                    json.dump(save_draft, _f, indent=4, ensure_ascii=False)
                print(f"[OK] Builder: sequence updated at '{edit_path}'.")
                saved_path = edit_path
            except OSError as exc:
                return jsonify({"error": str(exc)}), 500
        else:
            # New sequence (including one started from /api/clone_sequence,
            # which tags the draft with "_cloned_from" purely for the builder's
            # UI badge) — strip internal metadata before persisting.
            save_draft = {k: v for k, v in draft.items() if k != "_cloned_from"}
            try:
                saved_path = builder_service.save_sequence(save_draft, _SEQUENCES_DIR)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

        with state.lock:
            try:
                if state.factory is not None:
                    state.factory.reload_sequence(saved_path)
                else:
                    state.factory = AppFactory(_DEFAULT_VALUES_PATH, saved_path)
                state.controller = state.factory.create_inference_controller()
                state.factory.initialize_hardware()
                _sync_capture_resolution(state)
                state.current_sequence_path = saved_path
                state.current_mode = "inference"
                state.draft = None
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        return jsonify({"ok": True, "path": saved_path, "redirect": url_for("inspection")})

    # =========================================================================
    # Inspection API
    # =========================================================================

    @app.route("/api/sequences")
    def api_sequences():
        return jsonify(builder_service.list_sequences(_SEQUENCES_DIR))

    @app.route("/api/load_sequence_for_edit", methods=["POST"])
    def api_load_sequence_for_edit():
        """
        Open a saved sequence in the builder for editing.

        Copies the sequence JSON to the draft file and tags it with
        ``_edit_path`` so the builder's save button overwrites the original
        file rather than creating a new numbered one. Reinitialises hardware
        from the draft path. Rejected (409) while the loop is running.
        Body: {path: str}
        """
        data = request.get_json(force=True)
        path = data.get("path", "")
        if not path:
            return jsonify({"error": "Missing 'path'."}), 400
        if not os.path.isfile(path):
            return jsonify({"error": f"Sequence file not found: {path}"}), 404

        try:
            with open(path, "r", encoding="utf-8") as _f:
                draft = json.load(_f)
        except (json.JSONDecodeError, OSError) as exc:
            return jsonify({"error": f"Could not read sequence file: {exc}"}), 400

        # Tag so the builder knows to overwrite this file on save.
        draft["_edit_path"] = os.path.abspath(path)
        builder_service.save_draft(draft, _DRAFT_PATH)

        with state.lock:
            if state.is_busy:
                return jsonify({"error": "Stop the loop before editing a sequence."}), 409
            try:
                _load_sequence(state, _DRAFT_PATH)
                state.draft = draft
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        return jsonify({"redirect": url_for("builder")})

    @app.route("/api/clone_sequence", methods=["POST"])
    def api_clone_sequence():
        """
        Create a new sequence based on an existing one, without touching the
        original file.

        Unlike ``/api/load_sequence_for_edit`` (which tags the draft with
        ``_edit_path`` so the builder overwrites the source file), the clone
        draft is loaded WITHOUT that tag — the builder's normal Save Sequence
        button then treats it like any brand new sequence and writes a new
        auto-numbered ``sequence_NNN.json``, never overwriting the source.

        The source sequence's hardware config and pipeline/steps are copied
        verbatim (that is the point — reuse a similar part's setup instead of
        rebuilding it from scratch in the builder), but ``part_model`` and
        ``paths`` are regenerated for the new part: a clone is a different
        physical part and must never share images/model directories with the
        sequence it was cloned from.

        Body: {path: str, new_part_model: str}
        """
        data = request.get_json(force=True)
        path = data.get("path", "")
        new_part_model = (data.get("new_part_model") or "").strip()
        if not path:
            return jsonify({"error": "Missing 'path'."}), 400
        if not os.path.isfile(path):
            return jsonify({"error": f"Sequence file not found: {path}"}), 404
        if not new_part_model:
            return jsonify({"error": "Product model name is required."}), 400

        duplicate = builder_service.find_sequence_by_part_model(new_part_model, _SEQUENCES_DIR)
        if duplicate:
            return jsonify({
                "error": (
                    f"A sequence with model name '{new_part_model}' already exists "
                    f"({duplicate['filename']}). Use a different name."
                )
            }), 409

        try:
            with open(path, "r", encoding="utf-8") as _f:
                source = json.load(_f)
        except (json.JSONDecodeError, OSError) as exc:
            return jsonify({"error": f"Could not read sequence file: {exc}"}), 400

        draft = copy.deepcopy(source)
        draft.pop("_edit_path", None)  # Defensive: never present in a saved file, but a clone must never overwrite anything.
        draft["part_model"] = new_part_model
        draft["paths"] = _generate_paths_for_part_model(new_part_model)
        draft["_cloned_from"] = os.path.basename(path)  # UI badge only, stripped before saving.

        # Create all storage directories declared for the new part before
        # initialising hardware, same as /api/setup/finalize.
        for key in ("images_path", "inference_images_path",
                    "traceability_inference_path", "model_path"):
            dir_path = draft["paths"].get(key, "")
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

        builder_service.save_draft(draft, _DRAFT_PATH)

        with state.lock:
            if state.is_busy:
                return jsonify({"error": "Stop the loop before cloning a sequence."}), 409
            try:
                _load_sequence(state, _DRAFT_PATH)
                state.draft = draft
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        return jsonify({"redirect": url_for("builder")})

    @app.route("/api/resume_draft", methods=["POST"])
    def api_resume_draft():
        """
        Resume the builder from the last auto-saved draft.

        Useful after a page reload or server restart: reinitialises hardware
        from the draft file so the builder can capture preview frames.
        Rejected (409) while the loop is running.
        """
        if not os.path.isfile(_DRAFT_PATH):
            return jsonify({"error": "No draft found."}), 404

        draft = builder_service.load_draft(_DRAFT_PATH)
        if not draft.get("part_model"):
            return jsonify({"error": "Draft is empty. Start with the setup wizard."}), 400

        with state.lock:
            if state.is_busy:
                return jsonify({"error": "Stop the loop before resuming the draft."}), 409
            try:
                _load_sequence(state, _DRAFT_PATH)
                state.draft = draft
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        return jsonify({"redirect": url_for("builder")})

    @app.route("/api/load_sequence", methods=["POST"])
    def api_load_sequence():
        """
        Load a different sequence. Rejected (409) while the loop is running.
        Body: {path: str}
        """
        data = request.get_json(force=True)
        path = data.get("path", "")
        if not path:
            return jsonify({"error": "Missing 'path'."}), 400

        with state.lock:
            if state.is_busy:
                return jsonify({"error": "Stop the loop before changing sequence."}), 409
            try:
                _load_sequence(state, path)
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        return jsonify({"ok": True})

    @app.route("/api/status")
    def api_status():
        with state.lock:
            running   = state.is_running
            mode      = state.current_mode
            seq_path  = state.current_sequence_path
            ctrl      = state.controller
            factory   = state.factory

        cycle_count = 0
        forced_scrap_cycles = 0
        stopped_reason = None
        label       = None
        schedule    = None
        if ctrl is not None:
            if hasattr(ctrl, "get_status"):
                status = ctrl.get_status()
                cycle_count = status.get("cycle_count", 0)
                forced_scrap_cycles = status.get("forced_scrap_cycles", 0)
                stopped_reason = status.get("stopped_reason")
            if mode == "samples" and hasattr(ctrl, "_label"):
                label = ctrl._label
            if mode == "samples" and hasattr(ctrl, "get_schedule_status"):
                schedule = ctrl.get_schedule_status()

        return jsonify({
            "running":      running,
            "mode":         mode,
            "dry_run":      state.dry_run,
            "sequence":     os.path.basename(seq_path) if seq_path else None,
            "label":        label,
            "schedule":     schedule,
            "cycle_count":  cycle_count,
            "forced_scrap_cycles": forced_scrap_cycles,
            "stopped_reason": stopped_reason,
            "camera_ports": factory.get_camera_ports() if factory else [],
            "channel_status": factory.get_channel_status() if factory else {},
            "hardware_wedged_message": factory.get_hardware_wedged_message() if factory else None,
        })

    @app.route("/api/start", methods=["POST"])
    def api_start():
        with state.lock:
            if state.is_busy:
                return jsonify({"error": "Already running."}), 409
            if not state.has_sequence:
                return jsonify({"error": "No sequence loaded."}), 400
            ctrl     = state.controller
            mode     = state.current_mode
            seq_path = state.current_sequence_path
            # Clear manual preview frames so the inference loop's own frames
            # are served by api_frame instead of stale preview snapshots.
            state.captured_frames.clear()
        if ctrl is None:
            return jsonify({"error": "No controller initialized. Set a mode first."}), 400
        ctrl.start_loop()
        # Session-resume state is scoped to inference mode only — Samples mode
        # capture sessions are never auto-resumed (see _resume_session_if_any).
        if mode == "inference":
            _save_session_state(seq_path)
        else:
            _clear_session_state()
        return jsonify({"ok": True})

    @app.route("/api/stop", methods=["POST"])
    def api_stop():
        with state.lock:
            ctrl = state.controller
            state.controller_stopping = True
        if ctrl is not None:
            ctrl.stop_loop()
        with state.lock:
            state.controller_stopping = False
        # The operator stopped the loop on purpose — do not auto-resume it.
        _clear_session_state()
        return jsonify({"ok": True})

    @app.route("/api/system/restart_iris", methods=["POST"])
    def api_system_restart_iris():
        """
        Restart the Iris web service process.

        Relies on systemd's ``Restart=on-failure`` policy (see
        ``setup/iris.service``) to bring the process back up automatically
        within a few seconds — no sudo/root privileges needed, and no OS-level
        reboot. Rejected while the inspection/samples loop is running, since
        ``os._exit(1)`` would kill the process mid-cycle without a chance to
        release GPIO outputs cleanly.

        The PIN is NOT a real security boundary (this endpoint never touches
        the OS or a real credential) — it only guards against an accidental
        click, per the confirmed design note in .github/copilot-instructions.md.
        Configured via ``restart_pin`` in ``config/default_values.json`` (read
        fresh on every call, so editing the file takes effect without needing
        a restart first).

        Body: {pin: str}
        """
        data = request.get_json(force=True)
        try:
            with open(_DEFAULT_VALUES_PATH, "r", encoding="utf-8") as f:
                configured_pin = json.load(f).get("restart_pin", "")
        except (OSError, json.JSONDecodeError):
            configured_pin = ""
        if not configured_pin or data.get("pin") != configured_pin:
            return jsonify({"error": "Incorrect PIN."}), 403

        with state.lock:
            if state.is_busy:
                return jsonify({"error": "Stop the inspection/samples loop before restarting."}), 409

        def _delayed_exit():
            time.sleep(0.5)  # let the HTTP response flush before the process dies
            os._exit(1)

        threading.Thread(target=_delayed_exit, daemon=True, name="RestartIrisExit").start()
        return jsonify({"ok": True})

    @app.route("/api/set_mode", methods=["POST"])
    def api_set_mode():
        """
        Switch between inference and samples mode.
        Body: {mode: "inference" | "samples"}
        Rejected (409) while loop is running.
        """
        data = request.get_json(force=True)
        mode = data.get("mode", "")
        if mode not in ("inference", "samples"):
            return jsonify({"error": "mode must be 'inference' or 'samples'."}), 400

        with state.lock:
            if state.is_running:
                return jsonify({"error": "Stop the loop before changing mode."}), 409
            if not state.has_sequence:
                return jsonify({"error": "No sequence loaded."}), 400
            factory = state.factory
            try:
                if mode == "inference":
                    state.controller = factory.create_inference_controller()
                else:
                    state.controller = factory.create_samples_controller()
                state.current_mode = mode
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

        return jsonify({"ok": True, "mode": mode})

    @app.route("/api/set_label", methods=["POST"])
    def api_set_label():
        """
        Set the capture label (samples mode only).
        Body: {label: "ok" | "nok"}
        """
        data  = request.get_json(force=True)
        label = data.get("label", "ok")
        with state.lock:
            if state.current_mode != "samples":
                return jsonify({"error": "set_label is only valid in samples mode."}), 400
            ctrl = state.controller
        if ctrl is not None and hasattr(ctrl, "set_label"):
            ctrl.set_label(label)
        return jsonify({"ok": True, "label": label})

    @app.route("/api/schedule/enable", methods=["POST"])
    def api_schedule_enable():
        """
        Enable Schedule Timed Captures (samples mode only, Train OK / Test OK
        labels only — Test NOK is excluded since NOK occurrences are expected
        to be manually reviewed rather than auto-collected). In-memory only,
        same as ``dry_run`` — not persisted to disk, lost on server restart.

        Body: {images_per_window: int, interval_minutes: number, target_images: int}
        """
        data = request.get_json(force=True)
        try:
            images_per_window = int(data.get("images_per_window", 20))
            interval_minutes  = float(data.get("interval_minutes", 90))
            target_images     = int(data.get("target_images", 300))
        except (TypeError, ValueError):
            return jsonify({"error": "images_per_window, interval_minutes and target_images must be numbers."}), 400

        if images_per_window <= 0 or interval_minutes <= 0 or target_images <= 0:
            return jsonify({"error": "All schedule values must be positive."}), 400

        with state.lock:
            if state.current_mode != "samples":
                return jsonify({"error": "Schedule Timed Captures is only available in Samples mode."}), 400
            ctrl = state.controller

        if ctrl is None or not hasattr(ctrl, "enable_schedule"):
            return jsonify({"error": "No samples controller active."}), 400

        label = getattr(ctrl, "_label", "ok")
        if label not in ("ok", "test_ok"):
            return jsonify({"error": "Schedule Timed Captures is only available for the Train OK / Test OK labels."}), 400

        ctrl.enable_schedule(images_per_window, interval_minutes * 60, target_images)
        return jsonify({"ok": True, "schedule": ctrl.get_schedule_status()})

    @app.route("/api/schedule/disable", methods=["POST"])
    def api_schedule_disable():
        """Disable Schedule Timed Captures. Manual capture behavior resumes immediately."""
        with state.lock:
            ctrl = state.controller
        if ctrl is not None and hasattr(ctrl, "disable_schedule"):
            ctrl.disable_schedule()
        return jsonify({"ok": True})

    @app.route("/api/set_dry_run", methods=["POST"])
    def api_set_dry_run():
        """
        Enable or disable dry-run mode.
        NOK steps (step_number < 0) are skipped when dry-run is active.
        Body: {dry_run: true | false}
        Rejected (409) while loop is running.
        """
        data    = request.get_json(force=True)
        dry_run = bool(data.get("dry_run", False))

        with state.lock:
            if state.is_running:
                return jsonify({"error": "Stop the loop before changing dry-run mode."}), 409
            state.dry_run = dry_run
            ctrl = state.controller

        if ctrl is not None and hasattr(ctrl, "set_dry_run"):
            ctrl.set_dry_run(dry_run)

        return jsonify({"ok": True, "dry_run": dry_run})

    @app.route("/api/set_forced_scrap", methods=["POST"])
    def api_set_forced_scrap():
        """
        Enable or disable forced scrap mode.
        Forces the next N loops to SCRAP when active.
        Body: { "cycles": int } (e.g., 3 to enable for 3 cycles, 0 to disable immediately)
        Rejected (409) while loop is running.
        """
        data = request.get_json(silent=True) or {}
        cycles = data.get("cycles", 0) 

        with state.lock:
            if state.is_running:
                return jsonify({"error": "Stop the loop before changing forced scrap mode."}), 409
            ctrl = state.controller
        
        if ctrl is not None and hasattr(ctrl, "set_forced_scrap"):
            ctrl.set_forced_scrap(cycles)

        return jsonify({"ok": True, "forced_scrap_cycles": cycles})

    @app.route("/api/last_result")
    def api_last_result():
        with state.lock:
            ctrl = state.controller
        if ctrl is None:
            return jsonify(None)
        return jsonify(ctrl.get_last_result())

    @app.route("/api/capture_preview", methods=["POST"])
    def api_capture_preview():
        """
        On-demand preview capture for the inspection page.
        Only available while the loop is NOT running.
        """
        with state.lock:
            if state.is_running:
                return jsonify({"error": "Use the stream while the loop is running."}), 409
            if not state.has_sequence:
                return jsonify({"error": "No sequence loaded."}), 400
            factory = state.factory

        channels       = factory.get_camera_ports()
        errors: dict[str, str] = {}

        for channel in channels:
            try:
                frame = factory.capture_preview_frame(channel)
                _, jpeg = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
                )
                state.store_frame(channel, jpeg.tobytes())
            except Exception as exc:
                errors[channel] = str(exc)

        return jsonify({
            "captured": channels,
            "errors": errors,
            "hardware_wedged_message": factory.get_hardware_wedged_message(),
        })

    @app.route("/api/frame/<channel>")
    def api_frame(channel: str):
        """Return the last captured JPEG for a camera channel.

        Priority:
        1. Manual preview frames stored in ``IrisState`` (builder or on-demand preview).
        2. Last inference frames from the running controller (updated after each cycle).
        3. Disk fallback: ``images_path/latest/{view_name}.jpg``.
        4. 204 No Content if nothing is available.

        Always returns ``Cache-Control: no-store`` so Chromium (kiosk mode) never
        serves a stale cached response between inspection cycles.
        """
        _NO_CACHE = {"Cache-Control": "no-store"}

        jpeg = state.get_frame(channel)
        if jpeg is not None:
            return Response(jpeg, mimetype="image/jpeg", headers=_NO_CACHE)

        with state.lock:
            ctrl    = state.controller
            factory = state.factory

        if ctrl is not None and hasattr(ctrl, "get_last_inference_frames"):
            frames = ctrl.get_last_inference_frames()
            # captured_frames keyed by view_name (e.g. "section_1_A");
            # fall back to searching by channel suffix when direct lookup fails.
            frame = frames.get(channel)
            if frame is None:
                for vname, f in frames.items():
                    if vname.split("_")[-1] == channel:
                        frame = f
                        break
            if frame is not None:
                _, enc = cv2.imencode(
                    ".jpg",
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
                )
                return Response(enc.tobytes(), mimetype="image/jpeg", headers=_NO_CACHE)

        # Disk fallback: images_path/latest/{view_name}.jpg — persists across
        # server restarts. Written atomically by GuiInferenceAdapter after each cycle.
        if factory is not None:
            try:
                images_base = factory._sequence["paths"].get(
                    "images_path",
                    f"./data/images/{factory._sequence['part_model']}/",
                )
                latest_dir = os.path.join(images_base, "latest")
                if os.path.isdir(latest_dir):
                    for fname in os.listdir(latest_dir):
                        if not fname.endswith(".jpg"):
                            continue
                        view_name = fname[:-4]  # strip .jpg
                        if view_name.split("_")[-1] == channel:
                            bgr = cv2.imread(os.path.join(latest_dir, fname))
                            if bgr is not None:
                                _, enc = cv2.imencode(
                                    ".jpg", bgr,
                                    [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY],
                                )
                                return Response(enc.tobytes(), mimetype="image/jpeg", headers=_NO_CACHE)
            except Exception:
                pass

        return Response(status=204)

    @app.route("/api/heatmap/<path:view_name>")
    def api_heatmap(view_name: str):
        """Return an overlay heatmap PNG for the last inspection result.

        Composites the error map (JET colourmap, resized to match the original
        frame) over the captured frame: 55% original + 45% heatmap.  Falls back
        to a plain JET error-map image when no captured frame is available.
        """
        with state.lock:
            ctrl = state.controller
        if ctrl is None or not hasattr(ctrl, "_last_result"):
            return Response(status=204)

        part = ctrl._last_result
        if part is None:
            return Response(status=204)

        error_map = None
        for result in part.inspection_results:
            if result.view == view_name:
                error_map = result.error_map
                break

        if error_map is None:
            return Response(status=204)

        # Retrieve the original captured frame for this view (RGB numpy array).
        original_frame = None
        if hasattr(ctrl, "get_last_inference_frames"):
            frames = ctrl.get_last_inference_frames()
            original_frame = frames.get(view_name)
            if original_frame is None:
                channel = view_name.split("_")[-1]
                for vname, f in frames.items():
                    if vname.split("_")[-1] == channel:
                        original_frame = f
                        break

        # Normalise error map for colormap application.
        em_f32   = error_map.astype(np.float32)
        em_norm  = cv2.normalize(em_f32, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(em_norm, cv2.COLORMAP_JET)

        if original_frame is not None:
            # Apply masks → crop → resize so the background matches the error map space.
            with state.lock:
                factory = state.factory
            if factory is not None:
                try:
                    bg_frame = _preprocess_frame_for_overlay(
                        original_frame, view_name, factory._sequence
                    )
                except Exception:
                    bg_frame = original_frame
            else:
                bg_frame = original_frame
            # Resize heatmap to match the background frame dimensions.
            h, w       = bg_frame.shape[:2]
            hm_resized = cv2.resize(heatmap_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
            # Convert background RGB → BGR for OpenCV blend.
            orig_bgr   = cv2.cvtColor(bg_frame, cv2.COLOR_RGB2BGR)
            overlay    = cv2.addWeighted(orig_bgr, 0.55, hm_resized, 0.45, 0)
            out_img    = overlay
        else:
            out_img = heatmap_bgr

        _, png = cv2.imencode(".png", out_img)
        return Response(png.tobytes(), mimetype="image/png")

    # =========================================================================
    # Calibration page
    # =========================================================================

    @app.route("/calibration")
    def calibration():
        """Calibration page — capture → sweep → calibrate → deploy."""
        sequences = builder_service.list_sequences(_SEQUENCES_DIR)
        with state.lock:
            has_seq  = state.has_sequence
            seq_path = state.current_sequence_path
            factory  = state.factory
        view_sections: list[dict] = _build_view_sections(factory) if factory else []
        return render_template(
            "calibration.html",
            has_sequence=has_seq,
            sequence_path=seq_path,
            view_sections=view_sections,
            sequences=sequences,
            current_sequence=os.path.basename(seq_path) if seq_path else None,
        )

    # ── Calibration status ────────────────────────────────────────────────────

    @app.route("/api/calibration/status")
    def api_calibration_status():
        """Return current calibration task progress and capture target.

        If ``state.best_blocks`` is empty (e.g. after a server restart) and a
        sequence with a trained model is loaded, the blocks are auto-loaded from
        ``{model_path}/eval_config.json`` so the calibration page immediately
        shows the previous configuration without requiring a new sweep.
        """
        with state.lock:
            cal_running  = state.cal_capture_controller is not None and \
                           getattr(state.cal_capture_controller, "_running", False)
            fit_progress = state.calibration_progress or {}
            fit_running  = not fit_progress.get("done", True)
            if cal_running:
                mode = "capture"
            elif fit_running:
                mode = "sweep"
            else:
                mode = "idle"
            running     = cal_running or fit_running
            target      = state.calibration_target
            progress    = fit_progress
            # Auto-load best_blocks on first status poll after a server restart.
            # Priority: eval_config.json (post-calibration) → sweep_results.json (post-sweep).
            if not state.best_blocks and state.factory is not None:
                try:
                    model_path    = state.factory._sequence["paths"]["model_path"]
                    eval_cfg_path = os.path.join(model_path, "eval_config.json")
                    if os.path.isfile(eval_cfg_path):
                        with open(eval_cfg_path, "r") as f:
                            eval_cfg = json.load(f)
                        blocks = eval_cfg.get("blocks", {})
                        if blocks:
                            state.best_blocks = {v: int(b) for v, b in blocks.items()}
                    if not state.best_blocks:
                        sweep_path = os.path.join(model_path, "sweep_results.json")
                        if os.path.isfile(sweep_path):
                            with open(sweep_path, "r", encoding="utf-8") as f:
                                sweep_data = json.load(f)
                            blocks = sweep_data.get("best_blocks", {})
                            if blocks:
                                state.best_blocks = {v: int(b) for v, b in blocks.items()}
                            if not state.sweep_results:
                                state.sweep_results = sweep_data.get("sweep_results", [])
                except Exception:
                    pass  # Non-critical — page still works without stored blocks
            best_blocks = dict(state.best_blocks)
        return jsonify({
            "running":     running,
            "mode":        mode,
            "target":      target,
            "progress":    progress,
            "best_blocks": best_blocks,
        })

    @app.route("/api/calibration/sweep_results")
    def api_calibration_sweep_results():
        """Return the latest sweep results list (serializable dicts).

        If the in-memory list is empty (e.g. after a server restart) the
        results are auto-loaded from ``{model_path}/sweep_results.json``.
        """
        with state.lock:
            results = list(state.sweep_results)
            # Auto-load from disk when RAM is empty.
            if not results and state.factory is not None:
                try:
                    _model_path  = state.factory._sequence["paths"]["model_path"]
                    _sweep_path  = os.path.join(_model_path, "sweep_results.json")
                    if os.path.isfile(_sweep_path):
                        with open(_sweep_path, "r", encoding="utf-8") as _f:
                            _data = json.load(_f)
                        results = _data.get("sweep_results", [])
                        if results:
                            state.sweep_results = results
                            if not state.best_blocks:
                                _bb = _data.get("best_blocks", {})
                                if _bb:
                                    state.best_blocks = {v: int(b) for v, b in _bb.items()}
                except Exception:
                    pass
        return jsonify({"sweep_results": results})

    @app.route("/api/calibration/calibration_eval")
    def api_calibration_calibration_eval():
        """Return the per-view results of the last completed calibration.

        Reads ``{model_path}/calibration_eval.json``.  Returns an empty list
        if no calibration has been run yet.
        """
        with state.lock:
            factory = state.factory

        results = []
        if factory is not None:
            model_path = factory._sequence["paths"]["model_path"]
            eval_path  = os.path.join(model_path, "calibration_eval.json")
            if os.path.isfile(eval_path):
                try:
                    with open(eval_path, "r", encoding="utf-8") as _f:
                        results = json.load(_f).get("results", [])
                except Exception:
                    pass

        return jsonify({"results": results})

    @app.route("/api/calibration/image_counts")
    def api_calibration_image_counts():
        """
        Return the image count per set (train_ok, test_ok, test_nok) per view.
        """
        with state.lock:
            factory = state.factory

        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        try:
            images_base = factory._sequence["paths"].get(
                "images_path",
                f"./data/images/{factory._sequence['part_model']}/",
            )
        except (KeyError, AttributeError):
            return jsonify({"error": "Cannot resolve images_path."}), 400

        dirs = {
            "train_ok":  os.path.join(images_base, "train", "OK"),
            "test_ok":   os.path.join(images_base, "test",  "OK"),
            "test_nok":  os.path.join(images_base, "test",  "NOK"),
            "discarded": os.path.join(images_base, "discarded"),
        }
        counts = {}
        for key, directory in dirs.items():
            count = 0
            if os.path.isdir(directory):
                for root, _, files in os.walk(directory):
                    count += sum(
                        1 for f in files
                        if f.lower().endswith((".jpg", ".jpeg", ".png"))
                    )
            counts[key] = count
        return jsonify(counts)

    @app.route("/api/calibration/last_frame/<path:view_name>")
    def api_calibration_last_frame(view_name: str):
        """
        Return the most recent captured image for a view directly from disk.

        Serves the raw captured JPEG without applying the preprocessing
        pipeline, resized to a display-friendly width (≤ 800 px).

        Query param:
            target (str): ``train_ok`` | ``test_ok`` | ``test_nok``.
                Defaults to the currently selected calibration target.
        """
        with state.lock:
            factory = state.factory
            current_target = state.calibration_target

        if factory is None:
            return Response(status=204)

        target = request.args.get("target", current_target)
        _target_dirs = {
            "train_ok": os.path.join("train", "OK"),
            "test_ok":  os.path.join("test",  "OK"),
            "test_nok": os.path.join("test",  "NOK"),
        }
        target_subpath = _target_dirs.get(target, _target_dirs["train_ok"])

        images_base = factory._sequence["paths"].get(
            "images_path",
            f"./data/images/{factory._sequence['part_model']}/",
        )

        # Channel is the last _-separated segment (e.g. "A" from "front_view_section_1_A").
        #channel = view_name.split("_")[-1]
        img_dir = os.path.join(images_base, target_subpath, view_name)

        if not os.path.isdir(img_dir):
            return Response(status=204)

        files = sorted(
            [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")],
            reverse=True,
        )
        if not files:
            return Response(status=204)

        bgr = cv2.imread(os.path.join(img_dir, files[0]))
        if bgr is None:
            return Response(status=204)

        # Resize to display width (≤ 800 px) keeping aspect ratio.
        h, w = bgr.shape[:2]
        max_w = 800
        if w > max_w:
            bgr = cv2.resize(bgr, (max_w, int(h * max_w / w)))

        _, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(jpeg.tobytes(), mimetype="image/jpeg")

    # ── Calibration capture ───────────────────────────────────────────────────

    @app.route("/api/calibration/set_target", methods=["POST"])
    def api_calibration_set_target():
        """Set the capture target: train_ok | test_ok | test_nok."""
        data   = request.get_json(force=True)
        target = data.get("target", "train_ok")
        if target not in ("train_ok", "test_ok", "test_nok"):
            return jsonify({"error": "Invalid target. Must be train_ok, test_ok, or test_nok."}), 400
        with state.lock:
            state.calibration_target = target
        return jsonify({"ok": True, "target": target})

    @app.route("/api/calibration/start_capture", methods=["POST"])
    def api_calibration_start_capture():
        """
        Start the GPIO-triggered calibration capture loop.

        Reuses the samples controller with the calibration image directory
        derived from ``calibration_target``.
        """
        with state.lock:
            if state.is_running:
                return jsonify({"error": "Stop the inspection loop first."}), 409
            factory = state.factory
            target  = state.calibration_target

        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        # Map calibration target to its subdirectory under images_path.
        label_map = {
            "train_ok":  os.path.join("train", "OK"),
            "test_ok":   os.path.join("test",  "OK"),
            "test_nok":  os.path.join("test",  "NOK"),
        }
        label_subpath = label_map[target]

        images_base = factory._sequence["paths"].get(
            "images_path",
            f"./data/images/{factory._sequence['part_model']}/",
        )
        capture_dir = os.path.join(images_base, label_subpath)
        os.makedirs(capture_dir, exist_ok=True)

        # Build a temporary samples controller pointing to the calibration capture dir.
        # The controller writes directly to capture_dir/{channel}/ via images_path.
        from app.src.adapters.input.GuiSamplesAdapter import GuiSamplesAdapter
        from app.src.core.services.SampleCaptureService import SampleCaptureService

        spotlight_pins  = factory._sequence["hardware"].get("spotlight_gpio_pins", [])
        #camera_channels = factory._sequence["hardware"]["camera_port"]
        sequence_steps= factory._sequence.get("steps", [])
        trigger_pin     = factory._sequence["hardware"]["trigger_input_pin"]
        timeout_ms      = 8000

        service = SampleCaptureService(
            camera=factory._camera,
            mux=factory._mux,
            gpio=factory._gpio,
            spotlight_pins=spotlight_pins,
            sequence_steps=sequence_steps,
            #camera_channels=camera_channels,
            trigger_pin=trigger_pin,
            trigger_timeout_ms=timeout_ms,
            images_path=capture_dir,
            preprocessing_params=factory._sequence.get("preprocessing_image_parameters", [])
        )
        capture_controller = GuiSamplesAdapter(service=service, camera=factory._camera)
        # Label is empty: _save_frame resolves to capture_dir/{channel}/ directly.
        capture_controller.set_label("")

        with state.lock:
            state.cal_capture_controller = capture_controller

        capture_controller.start_loop()
        return jsonify({"ok": True, "target": target, "dir": capture_dir})

    @app.route("/api/calibration/stop_capture", methods=["POST"])
    def api_calibration_stop_capture():
        """Stop the calibration capture loop."""
        with state.lock:
            capture_controller = state.cal_capture_controller
        if capture_controller is not None and hasattr(capture_controller, "stop_loop"):
            capture_controller.stop_loop()
        with state.lock:
            state.cal_capture_controller = None
        return jsonify({"ok": True})

    # ── Sweep ─────────────────────────────────────────────────────────────────

    @app.route("/api/calibration/run_sweep", methods=["POST"])
    def api_calibration_run_sweep():
        """Start the full block sweep (b3–b17) in a background thread."""
        with state.lock:
            if state.is_running:
                return jsonify({"error": "Stop the inspection loop first."}), 409
            if state.calibration_controller is not None and \
               getattr(state.calibration_controller, "is_running", False):
                return jsonify({"error": "A calibration task is already running."}), 409
            factory = state.factory

        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        data           = request.get_json(force=True) or {}
        params         = data.get("params", {})
        backbones_dir  = "data/models/backbones"

        view_names  = _build_view_names(factory)
        view_configs = _build_view_configs(factory, view_names)
        image_dirs   = factory.get_calibration_image_dirs(view_names)

        with state.lock:
            if state.calibration_controller is None:
                state.calibration_controller = factory.create_calibration_controller()
            cal_ctrl = state.calibration_controller

        def _on_sweep_done(results):
            serialized = _serialize_sweep_results(results)
            best_blocks = {s.view_name: s.best_block for s in results}
            with state.lock:
                state.sweep_results          = serialized
                state.best_blocks            = best_blocks
                state.calibration_progress   = cal_ctrl.get_progress()
            # Persist to disk so results survive a page/server refresh.
            try:
                _model_path = factory._sequence["paths"]["model_path"]
                os.makedirs(_model_path, exist_ok=True)
                _sweep_path = os.path.join(_model_path, "sweep_results.json")
                with open(_sweep_path, "w", encoding="utf-8") as _f:
                    json.dump({"sweep_results": serialized, "best_blocks": best_blocks}, _f, indent=4)
            except Exception as _exc:
                print(f"[WARN] Could not save sweep_results.json: {_exc}")

        cal_ctrl.start_sweep(
            view_configs=view_configs,
            image_dirs=image_dirs,
            backbones_dir=backbones_dir,
            params=params,
            done_cb=_on_sweep_done,
        )

        with state.lock:
            state.calibration_progress = cal_ctrl.get_progress()

        return jsonify({"ok": True})

    # ── Cancel sweep ──────────────────────────────────────────────────────────

    @app.route("/api/calibration/cancel", methods=["POST"])
    def api_calibration_cancel():
        """Request cancellation of the active sweep. No-op if nothing is running."""
        with state.lock:
            cal_ctrl = state.calibration_controller
        if cal_ctrl is not None:
            cal_ctrl.cancel()
        return jsonify({"ok": True})

    # ── Final calibration ─────────────────────────────────────────────────────

    @app.route("/api/calibration/run_calibration", methods=["POST"])
    def api_calibration_run_calibration():
        """Start the final calibration using the best block per view from the sweep."""
        with state.lock:
            if state.is_running:
                return jsonify({"error": "Stop the inspection loop first."}), 409
            if state.calibration_controller is not None and \
               getattr(state.calibration_controller, "is_running", False):
                return jsonify({"error": "A calibration task is already running."}), 409
            factory    = state.factory
            best_blocks = dict(state.best_blocks)

        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        if not best_blocks:
            return jsonify({
                "error": "Run the sweep first to determine the best block per view."
            }), 400

        data          = request.get_json(force=True) or {}
        params        = data.get("params", {})
        backbones_dir = "data/models/backbones"
        model_path    = factory._sequence["paths"]["model_path"]

        view_names   = _build_view_names(factory)
        view_configs  = _build_view_configs(factory, view_names)
        image_dirs    = factory.get_calibration_image_dirs(view_names)

        with state.lock:
            if state.calibration_controller is None:
                state.calibration_controller = factory.create_calibration_controller()
            cal_ctrl = state.calibration_controller

        def _on_calibration_done(results):
            _write_calibration_eval(results, factory._sequence["paths"]["model_path"])
            with state.lock:
                state.calibration_progress = cal_ctrl.get_progress()
                seq_path    = state.current_sequence_path
                factory_ref = state.factory
            # Reload outside the lock: initialize_hardware() can block for several
            # seconds (or indefinitely if libcamera stalls). Holding state.lock
            # during that call would freeze every request thread waiting for it.
            if factory_ref is not None and seq_path:
                try:
                    factory_ref.reload_sequence(seq_path)
                    new_ctrl = factory_ref.create_inference_controller()
                    factory_ref.initialize_hardware()
                    with state.lock:
                        state.controller = new_ctrl
                        _sync_capture_resolution(state)
                        state.current_sequence_path = seq_path
                        state.current_mode = "inference"
                except Exception as exc:
                    print(f"[ERROR] Failed to reload sequence after calibration: {exc}")

        cal_ctrl.start_calibration(
            view_configs=view_configs,
            image_dirs=image_dirs,
            backbones_dir=backbones_dir,
            blocks=best_blocks,
            model_path=model_path,
            params=params,
            done_cb=_on_calibration_done,
        )

        with state.lock:
            state.calibration_progress = cal_ctrl.get_progress()

        return jsonify({"ok": True, "blocks": best_blocks})


    @app.route("/api/calibration/review_analysis")
    def api_calibration_review_analysis():
        """Analyze traceability data for a given date.

        Query params:
            date (str): Date in ``YYYYMMDD`` format. Defaults to today.
        """
        from datetime import date as _date

        with state.lock:
            factory = state.factory

        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        date_str = request.args.get("date", _date.today().strftime("%Y%m%d"))

        try:
            svc      = factory.create_traceability_review_service()
            analysis = svc.analyze(date_str)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        def _safe_min(lst):
            return round(min(lst), 6) if lst else None

        def _safe_max(lst):
            return round(max(lst), 6) if lst else None

        view_stats_payload = []
        for vs in analysis.view_stats:
            view_stats_payload.append({
                "view_name":        vs.view_name,
                "total_parts":      vs.total_parts,
                "nok_count":        vs.nok_count,
                "nok_rate":         round(vs.nok_rate, 4),
                "score_min":        _safe_min(vs.all_scores),
                "score_max":        _safe_max(vs.all_scores),
                "nok_score_min":    _safe_min(vs.nok_scores),
                "nok_score_max":    _safe_max(vs.nok_scores),
                "threshold_min":    round(vs.threshold_min, 6),
                "threshold_max":    round(vs.threshold_max, 6),
                "available_images": vs.available_images,
                "drift_pattern":    vs.drift_pattern,
            })

        return jsonify({
            "date_str":              analysis.date_str,
            "total_parts":           analysis.total_parts,
            "total_nok_parts":       analysis.total_nok_parts,
            "global_drift_detected": analysis.global_drift_detected,
            "jsonl_path":            analysis.jsonl_path,
            "view_stats":            view_stats_payload,
        })

    @app.route("/api/calibration/review_images")
    def api_calibration_review_images():
        """Return a paginated list of NOK image metadata for a view and date.

        Query params:
            date (str): Date in ``YYYYMMDD`` format.
            view_name (str): View name to filter on.
            page (int): 1-based page number. Defaults to 1.
            page_size (int): Items per page. Defaults to 20.

        Returns JSON:
            {"items": [{filename, score, time_str}], "total": int, "page": int, "page_size": int}
        """
        with state.lock:
            factory = state.factory
        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        date_str  = request.args.get("date", "")
        view_name = request.args.get("view_name", "")
        if not date_str or not view_name:
            return jsonify({"error": "'date' and 'view_name' query params are required."}), 400

        try:
            page      = max(1, int(request.args.get("page", 1)))
            page_size = max(1, min(100, int(request.args.get("page_size", 20))))
        except (ValueError, TypeError):
            return jsonify({"error": "'page' and 'page_size' must be integers."}), 400

        try:
            svc          = factory.create_traceability_review_service()
            items, total = svc.list_nok_images(date_str, view_name, page, page_size)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify({"items": items, "total": total, "page": page, "page_size": page_size})

    @app.route("/api/calibration/nok_images_all")
    def api_calibration_nok_images_all():
        """Return a paginated list of NOK inference images for a view, across
        every date in the traceability history (no date filter).

        Backs the "View Production NOK" viewer in Step 1 of the calibration
        page — lets the operator browse (and delete) NOK inference images
        without going through the date-scoped "Recalibrate from Production"
        analysis flow first.

        Query params:
            view_name (str): View name to filter on.
            page (int): 1-based page number. Defaults to 1.
            page_size (int): Items per page. Defaults to 20.

        Returns JSON:
            {"items": [{filename, abs_path, score, date_str, time_str}], "total": int, "page": int, "page_size": int}
        """
        with state.lock:
            factory = state.factory
        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        view_name = request.args.get("view_name", "")
        if not view_name:
            return jsonify({"error": "'view_name' query param is required."}), 400

        try:
            page      = max(1, int(request.args.get("page", 1)))
            page_size = max(1, min(100, int(request.args.get("page_size", 20))))
        except (ValueError, TypeError):
            return jsonify({"error": "'page' and 'page_size' must be integers."}), 400

        try:
            svc          = factory.create_traceability_review_service()
            items, total = svc.list_all_nok_images_for_view(view_name, page, page_size)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify({"items": items, "total": total, "page": page, "page_size": page_size})

    @app.route("/api/calibration/delete_nok_images", methods=["POST"])
    def api_calibration_delete_nok_images():
        """Permanently delete inference images from disk (disk-space cleanup).

        Body: {paths: [str, ...]}

        Destructive and irreversible — the UI only reaches this after an
        explicit operator confirmation modal (mirrors ``/api/review/delete``,
        which guards the calibration image tree the same way this guards
        ``inference_images_path``).
        """
        with state.lock:
            factory = state.factory
        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        data  = request.get_json(force=True)
        paths = data.get("paths", [])
        if not paths:
            return jsonify({"error": "'paths' must be a non-empty list."}), 400

        svc    = factory.create_traceability_review_service()
        result = svc.delete_nok_images(paths)
        return jsonify({"ok": True, **result})

    @app.route("/api/image_file")
    def api_image_file():
        """Serve an image file by its absolute path.

        The path is validated against the project's ``data/`` directory to
        prevent directory-traversal attacks.  Only files whose resolved
        absolute path starts with the absolute path of the ``data/`` folder
        are served.

        Query params:
            path (str): Absolute path to the image file.
        """
        rel_path = request.args.get("path", "")
        if not rel_path:
            return Response("Missing 'path' parameter.", status=400)

        # Resolve to an absolute path and validate it is inside data/.
        try:
            abs_path  = os.path.realpath(rel_path)
            data_root = os.path.realpath("data")
        except Exception:
            return Response("Invalid path.", status=400)

        if not abs_path.startswith(data_root + os.sep):
            return Response("Access denied.", status=403)

        if not os.path.isfile(abs_path):
            return Response("File not found.", status=404)

        return send_file(abs_path, mimetype="image/jpeg")

    # ── Step 1 review/relabel gallery (calibration page) ────────────────────
    #   Not to be confused with /api/calibration/review_images (NOK-production
    #   traceability review used by "Recalibrate from Production").

    _REVIEW_SETS = ("train_ok", "test_ok", "test_nok", "discarded")

    @app.route("/api/review/images")
    def api_review_images():
        """Return a paginated list of images for one (set, view) pair.

        Query params:
            set (str): One of 'train_ok', 'test_ok', 'test_nok', 'discarded'.
            view (str): View name.
            page (int): 1-based page number. Defaults to 1.
            page_size (int): Items per page. Defaults to 20.
        """
        with state.lock:
            factory = state.factory
        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        target_set = request.args.get("set", "")
        view_name  = request.args.get("view", "")
        if target_set not in _REVIEW_SETS:
            return jsonify({"error": f"'set' must be one of {_REVIEW_SETS}."}), 400
        if not view_name:
            return jsonify({"error": "'view' query param is required."}), 400

        try:
            page      = max(1, int(request.args.get("page", 1)))
            page_size = max(1, min(200, int(request.args.get("page_size", 20))))
        except (ValueError, TypeError):
            return jsonify({"error": "'page' and 'page_size' must be integers."}), 400

        svc = factory.create_capture_review_service()
        items, total = svc.list_images(target_set, view_name, page, page_size)
        return jsonify({"items": items, "total": total, "page": page, "page_size": page_size})

    @app.route("/api/review/relabel", methods=["POST"])
    def api_review_relabel():
        """Move a batch of images to a different set (discard or restore).

        Body: {paths: [str, ...], target_set: 'train_ok'|'test_ok'|'test_nok'|'discarded'}
        """
        with state.lock:
            factory = state.factory
        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        data       = request.get_json(force=True)
        paths      = data.get("paths", [])
        target_set = data.get("target_set", "")
        if target_set not in _REVIEW_SETS:
            return jsonify({"error": f"'target_set' must be one of {_REVIEW_SETS}."}), 400
        if not paths:
            return jsonify({"error": "'paths' must be a non-empty list."}), 400

        svc    = factory.create_capture_review_service()
        result = svc.relabel_images(paths, target_set)
        return jsonify({"ok": True, **result})

    @app.route("/api/review/delete", methods=["POST"])
    def api_review_delete():
        """Permanently delete a batch of images from disk.

        Destructive and irreversible — the UI only reaches this after an
        explicit operator confirmation modal (single image or bulk).

        Body: {paths: [str, ...]}
        """
        with state.lock:
            factory = state.factory
        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        data  = request.get_json(force=True)
        paths = data.get("paths", [])
        if not paths:
            return jsonify({"error": "'paths' must be a non-empty list."}), 400

        svc    = factory.create_capture_review_service()
        result = svc.delete_images(paths)
        return jsonify({"ok": True, **result})

    @app.route("/api/calibration/promote_and_recalibrate", methods=["POST"])
    def api_calibration_promote_and_recalibrate():
        """Promote NOK inference images to training, then start calibration.

        Expects JSON body:
            {"date_str": "20260616", "view_names": [...], "test_count": 5, "params": {},
             "excluded_images": ["20260616154207_section_1_A.jpg", ...]}
        """
        with state.lock:
            if state.is_running:
                return jsonify({"error": "Stop the inspection loop first."}), 409
            if state.calibration_controller is not None and \
               getattr(state.calibration_controller, "is_running", False):
                return jsonify({"error": "A calibration task is already running."}), 409
            factory     = state.factory
            best_blocks = dict(state.best_blocks)

        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400
        if not best_blocks:
            return jsonify({"error": "Run the sweep first (Step 2) before re-calibrating."}), 400

        data        = request.get_json(force=True) or {}
        date_str    = data.get("date_str", "")
        view_names  = data.get("view_names", [])
        test_count  = int(data.get("test_count", 5))
        train_count = int(data.get("train_count", 20))
        params      = data.get("params", {})
        excluded_images: set[str] = set(data.get("excluded_images", []))

        if not date_str or not view_names:
            return jsonify({"error": "date_str and view_names are required."}), 400

        try:
            svc      = factory.create_traceability_review_service()
            promoted = svc.promote_images(date_str, view_names, test_count, train_count,
                                          excluded_filenames=excluded_images)
        except Exception as exc:
            return jsonify({"error": f"Image promotion failed: {exc}"}), 500

        selected_blocks = {v: b for v, b in best_blocks.items() if v in view_names}
        if not selected_blocks:
            return jsonify({
                "ok": True,
                "promoted": promoted,
                "blocks": {},
                "warning": "No best-block configuration found. Run the sweep first.",
            })

        backbones_dir = "data/models/backbones"
        model_path    = factory._sequence["paths"]["model_path"]
        view_configs  = _build_view_configs(factory, view_names)
        image_dirs    = factory.get_calibration_image_dirs(view_names)

        with state.lock:
            if state.calibration_controller is None:
                state.calibration_controller = factory.create_calibration_controller()
            cal_ctrl = state.calibration_controller

        def _on_done(results):
            _write_calibration_eval(results, factory._sequence["paths"]["model_path"])
            with state.lock:
                state.calibration_progress = cal_ctrl.get_progress()
                seq_path    = state.current_sequence_path
                factory_ref = state.factory
            # Reload outside the lock — same reason as _on_calibration_done above.
            if factory_ref is not None and seq_path:
                try:
                    factory_ref.reload_sequence(seq_path)
                    new_ctrl = factory_ref.create_inference_controller()
                    factory_ref.initialize_hardware()
                    with state.lock:
                        state.controller = new_ctrl
                        _sync_capture_resolution(state)
                        state.current_sequence_path = seq_path
                        state.current_mode = "inference"
                except Exception as exc2:
                    print(f"[ERROR] Failed to reload sequence after recalibration: {exc2}")

        cal_ctrl.start_calibration(
            view_configs=view_configs,
            image_dirs=image_dirs,
            backbones_dir=backbones_dir,
            blocks=selected_blocks,
            model_path=model_path,
            params=params,
            done_cb=_on_done,
        )

        with state.lock:
            state.calibration_progress = cal_ctrl.get_progress()

        return jsonify({"ok": True, "promoted": promoted, "blocks": selected_blocks})

    @app.route("/api/calibration/export_traceability")
    def api_calibration_export_traceability():
        """Export traceability data for a given date as a tab-separated .txt file.

        The columns are discovered dynamically from all records in the JSONL so
        the export adapts automatically to any number of views or GPIO events.

        Query params:
            date (str): Date in ``YYYYMMDD`` format. Defaults to today.
        """
        import io
        import csv
        from datetime import date as _date

        with state.lock:
            factory = state.factory

        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        date_str = request.args.get("date", _date.today().strftime("%Y%m%d"))

        # Resolve the JSONL path through the existing review service.
        try:
            svc = factory.create_traceability_review_service()
            jsonl_path = svc.resolve_jsonl_path(date_str)
        except Exception as exc:
            return jsonify({"error": f"Could not resolve traceability path: {exc}"}), 500

        if not jsonl_path or not os.path.isfile(jsonl_path):
            return jsonify({"error": f"No traceability file found for {date_str}."}), 404

        # ── Parse all records ──────────────────────────────────────────────
        records = []
        with open(jsonl_path, "r", encoding="utf-8") as _f:
            for line in _f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not records:
            return jsonify({"error": "Traceability file is empty."}), 404

        # ── Discover all view names and trigger steps (in order of appearance) ─
        all_view_names    = []
        seen_views        = set()
        all_trigger_steps = []     # list of step_number values, ordered
        seen_steps        = set()

        for rec in records:
            for vr in rec.get("view_results", []):
                vn = vr.get("view_name", "")
                if vn and vn not in seen_views:
                    all_view_names.append(vn)
                    seen_views.add(vn)
            for te in rec.get("trigger_events", []):
                sn = te.get("step_number")
                if sn is not None and sn not in seen_steps:
                    all_trigger_steps.append(sn)
                    seen_steps.add(sn)

        all_trigger_steps.sort()

        # ── Build header ────────────────────────────────────────────────────
        base_cols = ["part_id", "model_id", "date_inspected", "duration_s",
                     "overall_status", "piece_detected"]

        trigger_cols = []
        for sn in all_trigger_steps:
            trigger_cols += [
                f"step{sn}_direction",
                f"step{sn}_pin",
                f"step{sn}_action",
                f"step{sn}_result",
            ]

        view_cols = []
        for vn in all_view_names:
            view_cols += [
                f"{vn}_classification",
                f"{vn}_score",
                f"{vn}_threshold_min",
                f"{vn}_threshold_max",
            ]

        header = base_cols + trigger_cols + view_cols

        # ── Build rows ──────────────────────────────────────────────────────
        rows = []
        for rec in records:
            part = rec.get("part", {})
            row  = {
                "part_id":        part.get("part_id", ""),
                "model_id":       part.get("model_id", ""),
                "date_inspected": part.get("date_inspected", ""),
                "duration_s":     part.get("duration_s", ""),
                "overall_status": part.get("overall_status", ""),
                "piece_detected": part.get("piece_detected", ""),
            }

            # Trigger events — index by step_number.
            trigger_map = {te["step_number"]: te for te in rec.get("trigger_events", [])}
            for sn in all_trigger_steps:
                te = trigger_map.get(sn, {})
                row[f"step{sn}_direction"] = te.get("direction", "")
                row[f"step{sn}_pin"]       = te.get("pin", "")
                row[f"step{sn}_action"]    = te.get("action", "")
                row[f"step{sn}_result"]    = te.get("result", "")

            # View results — index by view_name.
            view_map = {vr["view_name"]: vr for vr in rec.get("view_results", [])}
            for vn in all_view_names:
                vr = view_map.get(vn, {})
                row[f"{vn}_classification"]  = vr.get("classification", "")
                row[f"{vn}_score"]           = vr.get("score", "")
                row[f"{vn}_threshold_min"]   = vr.get("threshold_min", "")
                row[f"{vn}_threshold_max"]   = vr.get("threshold_max", "")

            rows.append(row)

        # ── Write TSV into memory ────────────────────────────────────────────
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=header, delimiter="\t",
                                lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

        tsv_bytes = buf.getvalue().encode("utf-8")
        buf.close()

        # Derive model id from first record for the filename.
        model_id  = records[0].get("part", {}).get("model_id", "model") if records else "model"
        filename  = f"traceability_{model_id}_{date_str}.txt"

        return send_file(
            io.BytesIO(tsv_bytes),
            mimetype="text/plain",
            as_attachment=True,
            download_name=filename,
        )

    @app.route("/api/calibration/promote_only", methods=["POST"])
    def api_calibration_promote_only():
        """Promote NOK inference images to training without starting calibration.

        Expects JSON body:
            {"date_str": "20260616", "view_names": [...], "test_count": 5,
             "train_count": 20, "excluded_images": [...]}
        """
        with state.lock:
            factory = state.factory

        if factory is None:
            return jsonify({"error": "No sequence loaded."}), 400

        data        = request.get_json(force=True) or {}
        date_str    = data.get("date_str", "")
        view_names  = data.get("view_names", [])
        test_count  = int(data.get("test_count", 5))
        train_count = int(data.get("train_count", 20))
        excluded_images: set[str] = set(data.get("excluded_images", []))

        if not date_str or not view_names:
            return jsonify({"error": "date_str and view_names are required."}), 400

        try:
            svc      = factory.create_traceability_review_service()
            promoted = svc.promote_images(date_str, view_names, test_count, train_count,
                                          excluded_filenames=excluded_images)
        except Exception as exc:
            return jsonify({"error": f"Image promotion failed: {exc}"}), 500

        return jsonify({"ok": True, "promoted": promoted})

    # =========================================================================
    # MJPEG stream
    # =========================================================================

    @app.route("/stream")
    def stream():
        """
        Lazy MJPEG stream. Active while the inspection loop OR the calibration
        capture loop is running. Returns 204 when neither is active.
        """
        with state.lock:
            active_controller = state.controller if state.is_running else None
            if active_controller is None:
                capture_controller = state.cal_capture_controller
                if capture_controller is not None and getattr(capture_controller, "_running", False):
                    active_controller = capture_controller
        if active_controller is None:
            return Response(status=204)
        return Response(
            _mjpeg_generator(active_controller, state),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    _resume_session_if_any(state)

    return app


# =============================================================================
# Helpers (module-level to avoid closure complexity)
# =============================================================================

def _preprocess_frame_for_overlay(
    frame_rgb: np.ndarray,
    view_name: str,
    sequence: dict,
) -> np.ndarray:
    """Apply mask, crop and resize pipeline to a frame for heatmap overlay alignment.

    Performs the same spatial transforms as ``SequenceSettings.preprocess_for_inference()``
    (canonical order: masks → crop → resize) but returns a uint8 RGB image so it can
    be used directly as the background for an overlay.  ``normalize_mobilenet`` is
    intentionally skipped.  If no matching pipeline entry is found, the original frame
    is returned unchanged.

    Args:
        frame_rgb: Full-resolution RGB uint8 frame from the camera.
        view_name: Canonical view name (``{prefix_view}_{channel}``).
        sequence: Parsed sequence JSON dict (``factory._sequence``).

    Returns:
        Preprocessed RGB uint8 image at ``resize_to_training_resolution`` dimensions
        (or the input frame if the pipeline cannot be resolved).
    """
    # Lookup pipeline by view name (exact match)
    pipelines = sequence.get("preprocessing_image_parameters", [])
    channel = view_name.split("_")[-1]
    
    pipeline_exact: dict[str, dict] = {}
    pipeline_by_port: dict[str, dict] = {}

    for entry in pipelines:
        view = entry.get("view", "")
        port = entry.get("camera_port", "")
        if view and port:
            exact_key = f"{view}_{port}"
            pipeline_exact[exact_key] = entry
        if port:
            pipeline_by_port[port] = entry

    entry = pipeline_exact.get(view_name)
    if not entry:
        entry = pipeline_by_port.get(channel, {})

    pipeline = entry.get("pipeline", [])

    if not pipeline:
        return frame_rgb

    ordered = sorted(
        pipeline,
        key=lambda t: SequenceSettings._TOOL_ORDER.get(t.get("tool", ""), 50),
    )

    img = frame_rgb.copy()
    for tool_def in ordered:
        tool = tool_def.get("tool", "")
        p    = tool_def.get("parameters", {})
        if tool == "put_black_circle":
            import cv2 as _cv2
            _cv2.circle(
                img,
                (int(p["x"]), int(p["y"])),
                int(p["radius"]),
                (0, 0, 0),
                thickness=-1,
            )
        elif tool == "put_black_rectangle":
            import cv2 as _cv2
            x, y = int(p["x"]), int(p["y"])
            _cv2.rectangle(
                img,
                (x, y),
                (x + int(p["w"]), y + int(p["h"])),
                (0, 0, 0),
                thickness=-1,
            )
        elif tool == "apply_roi_crop":
            x, y, w, h = int(p["x"]), int(p["y"]), int(p["w"]), int(p["h"])
            img = img[y : y + h, x : x + w]
        elif tool == "resize_to_training_resolution":
            import cv2 as _cv2
            img = _cv2.resize(
                img,
                (int(p["width"]), int(p["height"])),
                interpolation=_cv2.INTER_LINEAR,
            )
    return img


def _build_view_names(factory: "AppFactory") -> list[str]:
    """
    Flask-side wrapper over ``ViewConfigBuilder.build_view_names()`` — see
    that module for the canonical, Flask-free implementation shared with
    ``scripts/offload_calibration.py`` and IrisLink.

    Args:
        factory (AppFactory): Active factory with loaded sequence.

    Returns:
        list[str]: Ordered, deduplicated view names, e.g.
            ``["front_view_section_1_A"]``.
    """
    return build_view_names(factory._sequence)


def _build_view_configs(factory: "AppFactory", view_names: list[str]) -> list[dict]:
    """
    Flask-side wrapper over ``ViewConfigBuilder.build_view_configs()`` — see
    that module for the canonical, Flask-free implementation shared with
    ``scripts/offload_calibration.py`` and IrisLink.

    Args:
        factory (AppFactory): Active factory with loaded sequence.
        view_names (list[str]): View names to build configs for.

    Returns:
        list[dict]: Each entry has ``view_name``, ``masks``, ``roi``, and
            ``training_shape``.
    """
    return build_view_configs(factory._sequence, view_names)


def _build_view_sections(factory: "AppFactory") -> list[dict]:
    """
    Group view names by their ``prefix_view`` for display in the calibration
    page preview grid.

    Uses the canonical view names derived from ``camera_action`` steps (same
    source as ``_build_view_names``) so that ``<img id="cal-last-{view_name}">``
    elements always match the model file names produced by calibration.

    Args:
        factory (AppFactory): Active factory with a loaded sequence.

    Returns:
        list[dict]: Each entry has ``section_label`` (str, the ``prefix_view``)
            and ``views`` (list[str], view names in that group).
    """
    steps  = factory._sequence.get("steps", [])
    groups: dict[str, list[str]] = {}
    order:  list[str]            = []
    seen:   set[str]             = set()
    for step in sorted(steps, key=lambda s: s.get("step_number", 0)):
        if 0 <= step.get("step_number", -1) <= 999:
            for action in step.get("camera_action", []):
                prefix_view = action.get("prefix_view", "")
                camera_port = action.get("camera_port", "")
                if prefix_view and camera_port:
                    vn = f"{prefix_view}_{camera_port}"
                    if vn not in seen:
                        seen.add(vn)
                        if prefix_view not in groups:
                            groups[prefix_view] = []
                            order.append(prefix_view)
                        groups[prefix_view].append(vn)
    return [{"section_label": prefix, "views": groups[prefix]} for prefix in order]


def _write_calibration_eval(results: list, model_path: str) -> None:
    """Persist per-view calibration results to ``calibration_eval.json``.

    Called from both *Fit Model* and *Recalibrate from Production* callbacks
    so Step 3 always reflects the last completed calibration, independently of
    the sweep results shown in Step 2.

    Args:
        results: List of ``CalibrationResult`` objects returned by
            ``CalibrationService.run_calibration()``.
        model_path: Directory where ``calibration_eval.json`` is written.
    """
    from datetime import datetime as _dt

    serialized = []
    for r in results:
        try:
            serialized.append({
                "view_name":     r.view_name,
                "block":         r.block,
                "sep_ratio":     round(r.sep_ratio, 4),
                "auc":           round(r.auc, 4),
                "threshold_min": round(r.threshold_min, 6),
                "threshold_max": round(r.threshold_max, 6),
                "inference_type": r.inference_type,
            })
        except Exception:
            pass

    payload = {
        "timestamp": _dt.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results":   serialized,
    }
    eval_path = os.path.join(model_path, "calibration_eval.json")
    try:
        with open(eval_path, "w", encoding="utf-8") as _f:
            json.dump(payload, _f, indent=4)
    except Exception as exc:
        print(f"[WARN] Could not write calibration_eval.json: {exc}")


def _serialize_sweep_results(sweep_results: list) -> list[dict]:
    """
    Convert a list of ``SweepResult`` dataclass objects to serializable dicts.

    Args:
        sweep_results (list): List of ``SweepResult`` objects.

    Returns:
        list[dict]: JSON-serializable sweep result dicts.
    """
    serialized = []
    for sweep in sweep_results:
        block_dicts = []
        for br in sweep.block_results:
            block_dicts.append({
                "block":      br.block,
                "auc":        round(br.auc,       4),
                "sep_ratio":  round(br.sep_ratio, 3),
                "min_nok":    round(br.min_nok,   6),
                "max_ok":     round(br.max_ok,    6),
                "min_ok":     round(br.min_ok,    6),
                "mean_ok":    round(br.mean_ok,   6),
                "std_ok":     round(br.std_ok,    6),
                "cv_ok":      round(br.cv_ok,     4),
                "n_train_ok": br.n_train_ok,
                "n_test_ok":  br.n_test_ok,
                "n_test_nok": br.n_test_nok,
            })
        serialized.append({
            "view_name":     sweep.view_name,
            "block_results": block_dicts,
            "best_block":    sweep.best_block,
            "best_sep_ratio": round(sweep.best_sep_ratio, 3),
            "best_auc":      round(sweep.best_auc, 4),
            "best_cv_ok":    round(sweep.best_cv_ok, 4),
        })
    return serialized


def _generate_paths_for_part_model(part_model: str) -> dict:
    """
    Build the standard ``paths`` block for a given ``part_model``.

    Shared by ``/api/setup/finalize`` (brand new sequence) and
    ``/api/clone_sequence`` (new sequence based on an existing one) so both
    always agree on the same directory-naming convention — a clone must get
    its own images/model directories, never share them with the source
    sequence it was cloned from.

    Args:
        part_model (str): Product model name (already stripped/validated).

    Returns:
        dict: ``paths`` block ready to assign to a draft/sequence.
    """
    return {
        "images_path": f"./data/images/{part_model}/",
        "target_train_images": 1500,
        "inference_images_path": f"./data/images/{part_model}/inference/",
        "traceability_inference_path": f"./data/traceability/{part_model}/inference/",
        "model_path": f"./data/models/{part_model}/",
    }


def _load_sequence(state: IrisState, path: str, warmup: bool = False) -> None:
    """
    (Re)load a sequence into state.factory and reset the controller.

    Must be called with ``state.lock`` held by the caller.

    Args:
        state (IrisState): Application state.
        path (str): Path to the sequence JSON file.
        warmup (bool): If True (default False), runs ``warmup_all_channels()`` right
            after hardware init so the UI has fresh per-channel status as soon
            as an operator loads a sequence. Set to False for the automatic
            startup/resume-session reloads, so the real per-channel state left
            by ``scripts/cold_boot_camera_prewarm.py`` stays observable instead
            of being immediately overwritten.

    Raises:
        Exception: Any exception from AppFactory construction or hardware
            initialization is propagated to the caller.
    """
    if state.factory is None:
        state.factory = AppFactory(_DEFAULT_VALUES_PATH, path)
    else:
        state.factory.reload_sequence(path)

    state.controller = state.factory.create_inference_controller()
    state.factory.initialize_hardware()
    if warmup:
        state.factory.warmup_all_channels()
    _sync_capture_resolution(state)
    state.current_sequence_path = path
    state.current_mode = "inference"


def _save_session_state(sequence_path: str) -> None:
    """
    Persist that inference mode is actively running for ``sequence_path``, so
    ``_resume_session_if_any`` can bring it back automatically after a crash,
    a systemd restart, or the "Restart Iris" button. Best-effort: a failure to
    write this file must never break Start/Stop.
    """
    try:
        with open(_SESSION_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump({"sequence_path": sequence_path}, f)
    except OSError as exc:
        print(f"[WARN] Could not persist session state: {exc}")


def _clear_session_state() -> None:
    """Remove the persisted session-resume file, if any (best-effort)."""
    try:
        if os.path.exists(_SESSION_STATE_PATH):
            os.remove(_SESSION_STATE_PATH)
    except OSError as exc:
        print(f"[WARN] Could not clear session state: {exc}")


def _resume_session_if_any(state: IrisState) -> None:
    """
    Auto-resume a previously running inference session after the process
    restarts (crash, systemd ``Restart=on-failure``, or the "Restart Iris"
    button) — so an unattended production line does not stay halted just
    because the web service had to restart.

    Scoped to inference mode only: a Samples-mode capture session is never
    auto-resumed, since choosing which label to resume under is an operator
    decision that should not be made silently.
    """
    if not os.path.exists(_SESSION_STATE_PATH):
        return
    try:
        with open(_SESSION_STATE_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        seq_path = saved.get("sequence_path")
        if not seq_path or not os.path.isfile(seq_path):
            _clear_session_state()
            return

        with state.lock:
            _load_sequence(state, seq_path, warmup=False)
            state.controller.start_loop()
        print(f"[INFO] Auto-resumed inference session for '{seq_path}' after restart.")
    except Exception as exc:
        print(f"[WARN] Could not auto-resume previous session: {exc}")


def _sync_capture_resolution(state: IrisState) -> None:
    """
    Reconcile the capture resolution stored in the sequence / draft with the
    actual resolution the camera driver accepted.

    USB cameras (V4L2) silently round the requested resolution to the nearest
    supported sensor mode. After ``initialize_hardware()`` the adapter holds the
    real dimensions in ``_capture_resolution``. This function updates:

    * ``state.factory._sequence["hardware"]["camera_capture_resolution"]`` — so
      ``_build_view_configs()`` and the calibration pipeline use correct coords.
    * ``state.draft["hardware"]["camera_capture_resolution"]`` (if a draft is
      active) — so the builder canvas JS receives the corrected resolution via
      the DRAFT template variable and all ROI scale factors are accurate.

    The on-disk sequence / draft files are intentionally **not** rewritten: the
    stored value reflects what the user configured; the in-memory correction is
    applied automatically on every hardware init.
    """
    if state.factory is None:
        return
    actual_w, actual_h = state.factory.get_capture_resolution()
    hw = state.factory._sequence.get("hardware", {})
    stored = hw.get("camera_capture_resolution", [actual_w, actual_h])
    if [actual_w, actual_h] != stored:
        hw["camera_capture_resolution"] = [actual_w, actual_h]
        print(
            f"[INFO] _sync_capture_resolution: corrected "
            f"{stored[0]}×{stored[1]} → {actual_w}×{actual_h} in sequence."
        )
    if state.draft is not None:
        draft_hw = state.draft.get("hardware", {})
        draft_stored = draft_hw.get("camera_capture_resolution", [actual_w, actual_h])
        if [actual_w, actual_h] != draft_stored:
            draft_hw["camera_capture_resolution"] = [actual_w, actual_h]
            print(
                f"[INFO] _sync_capture_resolution: corrected "
                f"{draft_stored[0]}×{draft_stored[1]} → {actual_w}×{actual_h} in draft."
            )


def _camera_keepalive(state: IrisState, interval_s: float = 300.0) -> None:
    """
    Daemon thread: reads one preview frame every ``interval_s`` seconds while
    all loops are idle.

    Prevents the libcamera ISP pipeline from stalling after extended periods
    of inactivity. When the inspection loop is running it already keeps the
    camera continuously busy; the keepalive only acts when everything is idle.
    Calibration capture and fit tasks are excluded from the keepalive window
    to avoid interfering with MUX channel switching.

    Args:
        state (IrisState): Shared application state.
        interval_s (float): Seconds between keepalive reads. Defaults to 300 (5 min).
    """
    while True:
        time.sleep(interval_s)
        with state.lock:
            active = (
                state.is_running
                or (state.cal_capture_controller is not None
                    and getattr(state.cal_capture_controller, "_running", False))
                or bool(state.calibration_progress
                        and not state.calibration_progress.get("done", True))
            )
            factory = state.factory if not active else None
        if factory is None:
            continue
        ports = factory.get_camera_ports()
        if not ports:
            continue
        try:
            factory.capture_preview_frame(ports[0])
        except Exception:
            pass  # Failures are handled by the next real operation or crash recovery.


def _cycle_watchdog(state: IrisState, poll_interval_s: float = 5.0) -> None:
    """
    Daemon thread: force-restarts the process if the active inspection cycle
    has been running (since its trigger was received) longer than
    ``cycle_watchdog_timeout_s`` (config/default_values.json, default 60 s).

    Protects against a cycle hung inside a blocking call that never raises a
    Python exception and never checks a threading event (e.g. a frozen
    ``capture_request()`` inside libcamera C code) — the only way to recover
    is to kill the process and let systemd's ``Restart=on-failure``
    (``setup/iris.service``) bring it back up within a few seconds. This is
    the last line of defense for a genuine hang; the manual "Restart Iris"
    button cannot help here since ``state.is_busy`` stays True for as long as
    the loop thread is alive, which a hang does not change.

    Idle time waiting for the next production trigger — which can
    legitimately last hours between runs, and may span more than one
    chained indefinite ``wait_for_input`` step (e.g. "Esperar inicio de
    ciclo" then "Esperar pieza 1ra pos") — is deliberately NOT counted; see
    ``SequenceExecutor.get_live_cycle_age_s()``.

    ``cycle_watchdog_timeout_s`` is intentionally NOT exposed anywhere in the
    Iris web UI — it is only meant to be tuned by editing
    ``config/default_values.json`` directly, for the rare case where a real
    cycle legitimately needs more than the default 60 s.

    Args:
        state (IrisState): Shared application state.
        poll_interval_s (float): Seconds between checks. Defaults to 5.
    """
    while True:
        time.sleep(poll_interval_s)
        with state.lock:
            ctrl    = state.controller
            running = state.is_running
            mode    = state.current_mode
        if not running or mode != "inference" or ctrl is None:
            continue

        get_age = getattr(ctrl, "get_live_cycle_age_s", None)
        if get_age is None:
            continue
        age = get_age()
        if age is None:
            continue

        try:
            with open(_DEFAULT_VALUES_PATH, "r", encoding="utf-8") as f:
                timeout_s = json.load(f).get("cycle_watchdog_timeout_s", 60)
        except (OSError, json.JSONDecodeError):
            timeout_s = 60

        if age > timeout_s:
            print(
                f"[CRITICAL] Cycle watchdog: active cycle has been running for "
                f"{age:.1f}s (> {timeout_s}s timeout) — forcing process exit "
                f"so systemd can restart it."
            )
            os._exit(1)


def _mjpeg_generator(controller, state: IrisState):
    """
    MJPEG multipart generator. Runs only while ``state.is_running`` is True.

    FPS is capped at ``_STREAM_FPS`` to avoid saturating the network.
    The connection counter is tracked so future improvements can gate
    multi-client streaming limits.
    """
    global _stream_connections
    with _connection_lock:
        _stream_connections += 1
    try:
        while getattr(controller, "_running", False) or state.is_running:
            try:
                frame = controller.get_preview_frame()
            except Exception:
                time.sleep(1.0 / _STREAM_FPS)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(1.0 / _STREAM_FPS)
    finally:
        with _connection_lock:
            _stream_connections -= 1


if __name__ == "__main__":
    # Read logs_path from default_values so the logger uses the configured dir.
    try:
        with open(_DEFAULT_VALUES_PATH, "r", encoding="utf-8") as _cfg:
            _logs_path = json.load(_cfg).get("logs_path", "./logs")
    except Exception:
        _logs_path = "./logs"

    _logger = TimestampedFileLogger(_logs_path, suffix="iris")
    _logger.start()
    try:
        iris = create_iris_app()
        iris.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        _logger.stop()
