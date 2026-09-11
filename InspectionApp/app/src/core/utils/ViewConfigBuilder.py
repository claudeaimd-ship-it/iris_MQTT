"""
ViewConfigBuilder — pure helpers to derive view names and per-view calibration
configs from a parsed sequence JSON dict.

Extracted from ``IrisServer.py`` (2026-07-30) so this logic has no Flask or
hardware dependency and can be reused as-is by calibration tooling running on
a PC (``scripts/offload_calibration.py``, IrisLink) without hand-duplicating
it. Before this extraction, ``IrisServer.py`` and ``offload_calibration.py``
each carried their own copy of the same logic, and they had already drifted
apart (different function signatures, same behavior) — this module is now
the single source of truth for both.

Any change here changes view/ROI/mask resolution for both calibration and
inference — keep it in sync with ``SequenceSettings`` (canonical pipeline
tool execution order) and ``CalibrationService`` (consumer of the configs
built here).
"""
from __future__ import annotations


def build_view_names(sequence: dict) -> list[str]:
    """
    Derive view names from the camera_action steps in the sequence.

    The canonical ``view_name`` is ``{prefix_view}_{camera_port}`` as defined
    in each camera_action step — this is the same key that
    ``SequenceExecutor`` writes into ``captured_frames`` at runtime, so
    calibration file names and inference keys always match.

    Args:
        sequence (dict): Parsed sequence JSON (e.g. ``factory._sequence``).

    Returns:
        list[str]: Ordered, deduplicated view names, e.g.
            ``["front_view_section_1_A"]``.
    """
    steps: list[dict] = sequence.get("steps", [])
    seen: set[str] = set()
    view_names: list[str] = []
    for step in sorted(steps, key=lambda s: s.get("step_number", 0)):
        if 0 <= step.get("step_number", -1) <= 999:
            for action in step.get("camera_action", []):
                prefix_view = action.get("prefix_view", "")
                camera_port = action.get("camera_port", "")
                if prefix_view and camera_port:
                    vn = f"{prefix_view}_{camera_port}"
                    if vn not in seen:
                        seen.add(vn)
                        view_names.append(vn)
    return view_names


def build_view_configs(sequence: dict, view_names: list[str]) -> list[dict]:
    """
    Build the view config list required by ``CalibrationService``.

    For each view_name (derived from camera_action steps), the matching
    ``preprocessing_image_parameters`` entry is located by exact
    ``{view}_{camera_port}`` match first, falling back to ``camera_port``
    alone (the last ``_``-separated segment of the view_name). ROI, mask and
    resize parameters are extracted from that entry's pipeline. Defaults to
    the full capture frame when no matching entry or no ROI tool is found.

    Args:
        sequence (dict): Parsed sequence JSON (e.g. ``factory._sequence``).
        view_names (list[str]): View names to build configs for.

    Returns:
        list[dict]: Each entry has ``view_name``, ``masks``, ``roi``, and
            ``training_shape``.
    """
    pipelines    = sequence.get("preprocessing_image_parameters", [])
    cap_w, cap_h = sequence["hardware"].get("camera_capture_resolution", [640, 480])

    # Index pipeline entries by camera_port for O(1) lookup.
    pipeline_exact: dict[str, dict] = {}
    pipeline_by_port: dict[str, dict] = {}
    for entry in pipelines:
        view = entry.get("view", "")
        port = entry.get("camera_port", "")
        if view and port:
            pipeline_exact[f"{view}_{port}"] = entry
        if port:
            pipeline_by_port[port] = entry

    configs: list[dict] = []
    for view_name in view_names:
        channel = view_name.split("_")[-1]
        entry   = pipeline_exact.get(view_name) or pipeline_by_port.get(channel, {})
        pipeline = entry.get("pipeline", [])

        roi            = {"x": 0, "y": 0, "w": cap_w, "h": cap_h}
        training_shape = (cap_h, cap_w)
        masks: list[dict] = []

        for tool in pipeline:
            if tool.get("tool") == "apply_roi_crop":
                p   = tool.get("parameters", {})
                roi = {"x": p.get("x", 0), "y": p.get("y", 0),
                       "w": p.get("w", cap_w), "h": p.get("h", cap_h)}
            elif tool.get("tool") == "resize_to_training_resolution":
                p = tool.get("parameters", {})
                if p.get("width") and p.get("height"):
                    training_shape = (int(p["height"]), int(p["width"]))
            elif tool.get("tool") in ("put_black_circle", "put_black_rectangle"):
                masks.append(tool)

        configs.append({
            "view_name":      view_name,
            "masks":          masks,
            "roi":            roi,
            "training_shape": training_shape,
            "inference_type": entry.get("inference_type", "standard"),
        })
    return configs
