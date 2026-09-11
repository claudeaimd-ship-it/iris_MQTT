# InspectionApp — Copilot Instructions

**Author**: Alan Rojas — Team MDA & DA, Johnson Electric, Zacatecas, México

## What this project is

A Raspberry Pi 4B vision inspection system for industrial quality control.
It captures images from up to 4 CSI cameras through a hardware MUX, runs
split-model anomaly detection on a Coral USB Accelerator (EdgeTPU), classifies
each part as OK or NOK, and sends the result to a PLC via GPIO.

The codebase is a clean refactor of a working production system
(`Frambuesa/src/01-MuestrasPi_InferenciaPi/`) into hexagonal architecture.

Target hardware: Raspberry Pi 4B · RPi Camera Module v3 Wide NoIR · Coral USB
Accelerator · custom IO board (GPIO + I2C MUX) · industrial spotlights.

---

## Architecture: hexagonal (ports & adapters)

```
app/src/
├── interfaces/          ← ports (abstract contracts, no implementations)
│   ├── ICamera.py
│   ├── IControllableCamera.py        (extends ICamera: lens, exposure, restore)
│   ├── ICsiMux.py
│   ├── IGpio.py
│   ├── IInferenceEngine.py
│   ├── IRepository.py
│   └── IPaDiMFeatureExtractor.py     (load_backbone, extract_features — calibration only)
│
├── core/
│   ├── models/
│   │   └── Part.py              (domain model: Part, InspectionResult, TriggerEvent)
│   ├── services/
│   │   ├── InspectionService.py      (preprocessing + scoring logic)
│   │   ├── SampleCaptureService.py   (ad-hoc sample capture coordinator)
│   │   ├── CalibrationService.py     (PaDiM sweep + calibration, writes model files)
│   │   └── SequenceExecutor.py       (sequence JSON orchestrator)
│   ├── settings/
│   │   └── SequenceSettings.py  (typed access to pipelines, thresholds, scoring params)
│   └── utils/
│       ├── ImageProcessing.py         (clahe, roi_crop, circle mask, mobilenet norm…)
│       └── TimestampedFileLogger.py   (stdout/stderr → daily rotating log file)
│
├── adapters/
│   ├── output/   ← driven adapters (implement the ports)
│   │   ├── CsiCameraAdapter.py              (Picamera2)
│   │   ├── UsbCameraAdapter.py              (OpenCV VideoCapture)
│   │   ├── RpiCsiMuxAdapter.py              (GPIO + I2C channel switching)
│   │   ├── NoOpMuxAdapter.py                (null-object MUX for USB/IP/single-CSI setups)
│   │   ├── RpiGpioAdapter.py                (gpiozero; incl. `_interrupted` event + `interrupt()`/`reset_interrupt()`)
│   │   ├── NullGpioAdapter.py               (null-object GPIO for PC/Jetson; `interrupt()`/`reset_interrupt()` are no-ops)
│   │   ├── CoralUsbInferenceAdapter.py      (tflite_runtime + EdgeTPU delegate)
│   │   ├── CpuInferenceAdapter.py           (tflite_runtime or tensorflow, CPU inference)
│   │   ├── PaDiMInferenceAdapter.py         (onnxruntime, Gaussian anomaly scoring)
│   │   ├── LocalStorageAdapter.py           (JSON Lines + optional JPEG images)
│   │   └── PaDiMFeatureExtractorAdapter.py  (onnxruntime, implements IPaDiMFeatureExtractor)
│   └── input/    ← driving adapters (entry points, no ports defined)
│       ├── GuiInferenceAdapter.py    (production: background loop + crash recovery)
│       ├── GuiSamplesAdapter.py      (ad-hoc capture: thin adapter over SampleCaptureService)
│       └── GuiCalibrationAdapter.py  (calibration: background thread, progress reporting)
│
AppFactory.py  ← composition root — only class that knows concrete types
```

`AppFactory` is the **only** place that imports concrete adapter classes.
All other layers depend only on interfaces.

---

## Architecture invariants

These rules must hold after every change. Verify before proposing any refactoring.

1. **No adapter imports another adapter.** Adapters never depend on each other.
2. **`AppFactory` is the only file that imports concrete adapter classes.**
3. **`core/` depends only on `interfaces/` and other `core/` modules.** Never on `adapters/`.
4. **Input adapters (`adapters/input/`) receive ONE service from `core/services/`, never raw `ICamera`/`IGpio`/`ICsiMux`.**
   - `GuiInferenceAdapter` receives `SequenceExecutor`.
   - `GuiSamplesAdapter` receives `SampleCaptureService`.
   - `GuiCalibrationAdapter` receives `CalibrationService`.
   - The only extra dependency allowed is `ICamera` for MJPEG preview (read-only, no control).
5. **Services in `core/services/` depend only on interfaces, never on concrete adapters.**
6. **Do not add error handling for scenarios that cannot happen.** Only validate at system boundaries.
7. **Do not refactor beyond what is requested.** No incidental cleanup.

---

## Entry points

| File | Purpose |
|---|---|
| `main.py` | CLI test loop — runs one `SequenceExecutor.run()` per input |
| `GuiInferenceAdapter` | Production: background thread, crash recovery, MJPEG preview |
| `GuiSamplesAdapter` | Training: GPIO-triggered loop, labeled sample capture |

`main.py` wraps everything in `TimestampedFileLogger` so all output goes to both
the terminal and `logs/YYYYMM/YYYYMMDD_log-inspection.txt`.

---

## AppFactory usage pattern

```python
factory = AppFactory("config/default_values.json", "config/sequence_001.json")

# Production inference
controller = factory.create_inference_controller()
factory.initialize_hardware()   # must call AFTER create_*
try:
    controller.start_loop()
    ...
finally:
    factory.shutdown()

# Training sample capture
samples = factory.create_samples_controller()
factory.initialize_hardware()   # must call AFTER create_*
samples.set_label("ok")        # label written to disk (default: "ok")
samples.start_loop()            # blocks-free GPIO-triggered loop in background thread
...
samples.stop_loop()
factory.shutdown()

# CLI test
executor = factory.create_sequence_executor()
factory.initialize_hardware()
part = executor.run("20260518-0001")
```

`initialize_hardware()` calls `camera.initialize_camera()`.
`shutdown()` calls `close_camera()`, `mux.close()`, `gpio.close()`.

`AppFactory._build_hardware()` selects the MUX implementation based on `camera_type`
and `number_of_cameras` from the sequence JSON:

| Condition | MUX adapter used |
|---|---|
| `camera_type == "CSI"` and `number_of_cameras > 1` | `RpiCsiMuxAdapter` (real hardware) |
| any other type or single-camera CSI | `NoOpMuxAdapter` (null object, no-op) |

Services and input adapters always call `mux.select_channel()` unconditionally — they
never branch on the camera type. All branching lives exclusively in `AppFactory`.

---

## Configuration files

### `config/default_values.json`
Hardware defaults that apply to every sequence:
- `camera_capture_resolution` / `camera_preview_resolution`
- `csi_channels`: list of `{name, i2c_cmd, gpio_state}` for A/B/C/D
- `timeout_waiting_for_signal_ms`: fallback GPIO wait timeout
- `part_id_counter_path`: path to daily counter JSON for part ID generation
- `logs_path`, `inference_images_path`
- `restart_pin`: PIN for the topbar "Restart Iris" button (not a real credential, see "Reboot button" row below)
- `cycle_watchdog_timeout_s`: max seconds a triggered inspection cycle may run before the in-process cycle watchdog force-restarts the process (see "In-process cycle watchdog" row below). Intentionally has no UI — hand-edit this file only.

### `config/camera_catalog.json`
Catalog of supported camera models per connection type. Loaded by `AppFactory` at startup
and by `IrisServer` for the setup wizard.
Each entry has `name`, `sensor`, `supports_af_motor`, `setup_notes`, and `resolutions`.

`supports_af_motor` controls whether `CsiCameraAdapter` includes `AfMode` and `LensPosition`
in Picamera2 initialization controls, and whether `set_lens_position()` is a real call or a
safe no-op. Must be `false` for sensors without an electronic focus motor (e.g. imx477/HQ Camera,
imx219/v2). If the model is not found in the catalog, defaults to `true` with a `[WARN]`.

`resolutions` is a dict with three tiers (`high`, `medium`, `low`), each with `label`, `width`,
`height`. These map to the user-visible quality presets in the setup wizard (Alta/Media/Baja).
The `high` entry is always the sensor's maximum native resolution.

### `config/device_catalog.json`
Catalog of supported host platforms. Loaded by `IrisServer` and passed to the setup wizard.
Each entry has:
- `name` (str): display name shown in the wizard
- `device_type` (str): canonical key stored in sequence JSON and read by `AppFactory`
- `supports_gpio` (bool): controls whether step 3 (GPIO) is shown in the setup wizard
- `supported_camera_types` (list): camera types allowed for this device (e.g. `["CSI", "USB"]`)
- `supported_inference_devices` (list): inference options available (e.g. `["Coral USB Accelerator", "CPU"]`)
- `max_cameras` (int): maximum cameras the platform can drive

Current entries: `"PC"`, `"RaspberryPi"`, `"Jetson"`.

### `config/sequence_001.json`
Per-product configuration:
- `part_model`: string used in every traceability record (`"nissan_shroud"`)
- `paths`: `images_path`, `inference_images_path`, `traceability_inference_path`, `model_path`
- `hardware`: camera ports, GPIO pins, spotlight pins, `trigger_input_pin`, `camera_type`, `camera_model`, `number_of_cameras`
- `preprocessing_image_parameters`: list of per-view pipeline definitions
- `scoring`: `top_k_pixels` (overridden by `eval_config.json` if present)
- `steps`: ordered list of step objects (see Step format below)

`hardware.trigger_input_pin` is the GPIO input pin number for the PLC/robot start signal.
Used by both `GuiInferenceAdapter` (via `SequenceExecutor` step 1) and
`GuiSamplesAdapter` (via `SampleCaptureService.wait_for_trigger()`). Must match the pin
listed as `"type": "input"` in `hardware.gpio_configuration`.

### `{model_path}/eval_config.json`  ← generated by training pipeline
Per-model scoring parameters. Overrides `sequence.json` scoring block:
```json
{
    "top_k_pixels": 2,
    "border_crop_px": 1,
    "gaussian_sigma": 0.0,
    "use_clahe": false
}
```
Priority: **`eval_config.json` > `sequence.json [scoring]` > code defaults**

`use_clahe` controls whether CLAHE is applied to frames before feature extraction at **inference time**.
Must match the value used during calibration (`training_padim.py USE_CLAHE` or `CalibrationService` param).
Defaults to `false` if absent (backward-compatible).

`CoralUsbInferenceAdapter.load_models_from_directory()` loads this automatically.
`AppFactory.create_sequence_executor()` passes it to `SequenceSettings`.

**⚠ CRITICAL — partial recalibration rule**: `CalibrationService.run_calibration()` always reads the existing `eval_config.json`, merges `blocks` and `backbone_paths` (existing entries for untouched views are preserved, new entries override), and writes back the merged result. This is essential when recalibrating only a subset of views — the other views' backbone paths and block assignments must not be lost. If `eval_config.json` is accidentally corrupted (e.g., only one view remains), restore it by merging `sweep_results.json["best_blocks"]` with the recalibrated view's entry.

### `{model_path}/calibration_eval.json`  ← generated by IrisServer after every calibration
Per-view metrics of the **last completed calibration** (either Fit Model or Recalibrate from Production). Written by `_write_calibration_eval()` helper in `IrisServer.py`. Read by `GET /api/calibration/calibration_eval`. Displayed in the "Current model evaluation" panel in Step 3 of the calibration page.
```json
{
    "timestamp": "2026-06-16 15:32:00",
    "results": [
        {"view_name": "section_2_B", "block": 7, "sep_ratio": 1.3568, "auc": 0.9812, "threshold_min": 0.0021, "threshold_max": 0.4530}
    ]
}
```
This file is **independent of sweep_results.json** — it reflects the actual test-set performance of the fitted model, not the sweep's per-block candidates.

### `{model_path}/thresholds.json`  ← generated by training pipeline
Per-view acceptance thresholds:
```json
{
    "front_view_section_1_A": {"min": 0.0001, "max": 0.0050},
    "front_view_section_1_B": {"min": 0.0002, "max": 0.0060}
}
```
Keys must match `view_name` format: `{prefix_view}_{channel}`.

---

## Step format in sequence JSON

Steps are classified by `step_number`:

| Range | Class | Behavior |
|---|---|---|
| 0 – 999 | Normal | GPIO, camera, and detect_piece actions; run in order |
| ≥ 1001 | Inference | Calls `InspectionService.execute_full_inspection_from_frames()` |
| < 0 | NOK | GPIO actions; executed only if overall result is NOK **and** `dry_run=False` |

**GPIO action types**: `turn_on`, `turn_off`, `send_output`, `wait_for_input`

**Camera action**: `camera_port` + `prefix_view` → builds `view_name = {prefix_view}_{camera_port}`

Camera steps also support inline `set_time_exposure` and `set_lens_position`
in their action list; `SequenceSettings` silently skips them in the preprocessing
pipeline since they are handled by `SequenceExecutor`.

**Detect piece action** (`detect_piece_action`): captures a frame from the specified camera,
crops the ROI, and checks whether the **mean pixel brightness** (grayscale) of the ROI is
below `darkness_threshold`. The piece is black, so a present piece lowers the mean brightness.

- If `mean_brightness < darkness_threshold` → `part.piece_detected = True` → sequence continues.
- If `mean_brightness >= darkness_threshold` → `part.piece_detected = False` →
  `part.overall_status = False` → all remaining Normal and Inference steps are skipped →
  NOK steps execute (unless `dry_run=True`).

Only one `detect_piece_action` step is allowed per sequence.
ROI coordinates are in **capture resolution** (same space as all other canvas tools).
The ROI is drawn in the Iris builder using the "Detection ROI" tool (orange rectangle),
distinct from the preprocessing ROI (blue).

**Wait for piece action** (`wait_for_piece_action`): visual trigger — blocks the sequence in a
polling loop until a piece is detected, replacing a GPIO `wait_for_input` step when no PLC
wiring is available. Captures a reference frame at the start (empty fixture), then polls the
camera at `poll_interval_ms` intervals, computing the **mean absolute grayscale difference**
between the live ROI and the reference ROI. Triggers when the difference reaches
`pixel_diff_threshold`.

- Detection sets `part.piece_detected = True` and resets `part._actual_start_time`.
- `interrupt()` / `stop_loop()` exits the loop via `_stop_event` with a `TimeoutError`,
  handled identically to an interrupted `wait_for_input`.
- Polling uses `capture_frame()` on the `main` (full-resolution) stream — no stop/start,
  dual-stream already running. Recommended poll interval: 100–200 ms.
- Only one `wait_for_piece_action` step is allowed per sequence.
- Implemented in both `SequenceExecutor` (inference mode) and `SampleCaptureService`
  (samples mode) so it works the same way regardless of which adapter drives the cycle.
- ROI is drawn in the Iris builder with the same orange "Detection ROI" tool.
- Builder step type: **"Wait for Piece (visual trigger)"**.

Parameters:

| Key | Default | Description |
|---|---|---|
| `camera_port` | — | Camera channel to poll (e.g. `"A"`) |
| `roi` | — | `{x, y, w, h}` in capture-resolution pixel space |
| `pixel_diff_threshold` | `15` | Mean absolute grayscale difference (0–255) that triggers detection |
| `poll_interval_ms` | `100` | Milliseconds between camera polls (100 ms ≈ 10 fps) |
| `timeout_ms` | `0` | Max wait in ms; `0` = indefinite |
| `stabilization_ms` | `0` | Pause before capturing the reference frame (useful after MUX switch) |

Example step:
```json
{
    "step_number": 5,
    "description": "Wait for piece (visual trigger)",
    "wait_for_piece_action": {
        "camera_port": "A",
        "roi": {"x": 500, "y": 400, "w": 120, "h": 120},
        "pixel_diff_threshold": 15,
        "poll_interval_ms": 100,
        "timeout_ms": 0,
        "stabilization_ms": 0
    }
}
```

Example steps:
```json
{
    "step_number": 2,
    "description": "Detect piece presence — camera B.",
    "detect_piece_action": {
        "camera_port": "B",
        "roi": {"x": 100, "y": 200, "w": 150, "h": 150},
        "darkness_threshold": 80
    }
}
```
```json
{
    "step_number": 3,
    "description": "Capture front view — camera A.",
    "camera_action": [
        { "camera_port": "A", "prefix_view": "front_view_section_1" }
    ]
}
```

---

## view_name convention

`view_name` is always `{prefix_view}_{channel}`, e.g. `front_view_section_1_A`.
The last `_`-separated segment is always the camera channel.
This key is used consistently across:
- `captured_frames` dict in `SequenceExecutor`
- PaDiM model file names: `padim_{view_name}_params.npz`
- `thresholds.json` keys
- `InspectionResult.view`
- `LocalStorageAdapter` JSONL records

**Canonical source of `view_name`**: `prefix_view` from `camera_action` steps in `sequence.json`,
not `view` from `preprocessing_image_parameters`. These two fields may differ (the builder lets
them be set independently). `_build_view_names()` in `IrisServer.py` always reads from
`camera_action` steps to derive the view names used for calibration model files. This ensures
that calibration output (`padim_{view_name}_params.npz`) and runtime `captured_frames` keys
always match.

`_build_view_sections()` (calibration page preview grid) also derives view names from
`camera_action` steps so the `<img id="cal-last-{view_name}">` elements always match the
model file names. It must never use `preprocessing_image_parameters[].view` as a source.

`SequenceSettings.preprocess_for_inference()` resolves the preprocessing pipeline with 3-level
lookup: (1) exact `view_name` match, (2) prefix match (strip channel suffix), (3) port match —
finds any pipeline entry whose `camera_port` equals the channel suffix. This handles the case
where `prefix_view` in the step differs from `view` in `preprocessing_image_parameters`.

---

## Split inference pipeline

```
raw frame (RGB888 numpy)
    ↓  SequenceSettings.preprocess_for_inference()
preprocessed (1, H, W, 3) float32 MobileNetV2-normalized
    ↓  CoralUsbInferenceAdapter.predict()
    ├─ Teacher (MobileNetV2 float32, CPU) → features_original (1, H', W', C')
    └─ Student (autoencoder int8, EdgeTPU) → features_reconstructed (1, H', W', C')
    ↓  InspectionService._calculate_anomaly_score_and_error_map()
MSE over channel axis → 2D error_map
    ↓ optional Gaussian blur (settings.gaussian_sigma)
    ↓ border crop (settings.border_crop_px pixels)
score = mean(top-k highest pixel errors)  k = settings.top_k_pixels
    ↓
threshold_min ≤ score ≤ threshold_max → is_ok
```

Model file naming convention (must be exact):
```
teacher_{view_name}_float32.tflite
student_{view_name}_int8_edgetpu.tflite
```

The EdgeTPU delegate is loaded once as a class-level singleton in
`CoralUsbInferenceAdapter` and reused for all student models.

---

## Preprocessing pipeline tools

Defined per-view in `sequence.json` under `preprocessing_image_parameters`.
Each entry has `view`, `section`, `camera_port`, and `pipeline` (list of tools).

| tool key | function | required params | configured by |
|---|---|---|---|
| `apply_roi_crop` | crop rectangle | `x, y, h, w` | Iris builder (canvas) |
| `put_black_circle` | mask circle black | `x, y, radius` | Iris builder (canvas) |
| `put_black_rectangle` | mask rectangle black | `x, y, w, h` | Iris builder (canvas) |
| `resize_to_training_resolution` | cv2 resize | `width, height` | Iris builder (resize panel) |
| `normalize_mobilenet` | MobileNetV2 ImageNet norm | — | Teacher-Student only — NOT used for PaDiM |
| `set_time_exposure` | camera control (skipped here) | — | Iris builder (camera settings) |
| `set_lens_position` | camera control (skipped here) | — | Iris builder (camera settings) |

**Canonical tool execution order** (enforced by `SequenceSettings._TOOL_ORDER`, independent of JSON order):
1. `apply_clahe` — on full-resolution image
2. `put_black_circle` / `put_black_rectangle` — on full-resolution image, **before** crop
3. `apply_roi_crop`
4. `resize_to_training_resolution`
5. `normalize_mobilenet` (Teacher-Student only)

**MobileNetV2 normalisation — architecture difference:**
- **Teacher-Student (TFLite)**: normalisation is baked into the TFLite model during export. Pipeline JSON **must** include `normalize_mobilenet` (or normalisation can be in the export). Batch dimension `(1, H, W, C)` is always added by `SequenceSettings` as the final step.
- **PaDiM (ONNX)**: `PaDiMInferenceAdapter.predict()` normalises internally with `_normalize_mobilenet()` before running the backbone, matching what `PaDiMFeatureExtractorAdapter._preprocess()` does during calibration. Pipeline JSON must **NOT** include `normalize_mobilenet` for PaDiM sequences — it would have no effect and cause confusion. None of the existing PaDiM sequence JSONs include it.

**CLAHE is NOT a pipeline tool in the sequence JSON.** Whether CLAHE is applied
is decided by the training script (`USE_CLAHE` flag in `training_padim.py`). The
training script encodes the CLAHE decision in the calibrated `padim_*_params.npz`
(because the Gaussian statistics are fit on the CLAHE-transformed features). The
`apply_clahe` key is no longer emitted by the Iris builder and must not be added
back. `ImageProcessing.apply_clahe` still exists and can be called directly by
the training pipeline; it is simply not part of the inference-time preprocessing
pipeline.

`normalize_mobilenet` uses mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
Batch dimension `(1, H, W, C)` is always added as the final step by `SequenceSettings.preprocess_for_inference()` (after the pipeline loop).

**Masks must be applied before ROI crop.** `put_black_circle` and `put_black_rectangle` coordinates are always in the **full-resolution** image space. Applying them after `apply_roi_crop` would put them outside the cropped area for most real coordinates. `SequenceSettings._TOOL_ORDER` enforces the correct order regardless of JSON order. `CalibrationService._extract_all()` applies `_apply_masks()` before the crop for the same reason.

---

## Domain model

### `Part`
```python
Part(part_id: str, model_id: str)
    .date_inspected: datetime
    .time_inspected: float | None      # seconds, set at end of run()
    .inspection_results: list[InspectionResult]
    .triggers: list[TriggerEvent]
    .piece_detected: bool | None       # None = not checked; True = detected; False = absent
    .dry_run: bool                     # True if executed in dry-run mode (NOK steps skipped)
    .overall_status: bool              # False if empty or piece_detected=False; all(is_ok) otherwise
    .failed_channel: str | None        # camera channel that caused a system_error_paused abort, if any
    .failed_channel_error: str | None  # the error message from that channel's failure
```

### `InspectionResult`
```python
@dataclass
InspectionResult(view, is_ok, score, threshold_used: tuple[float,float], error_map)
```

### `TriggerEvent`
```python
@dataclass
TriggerEvent(step_number: int, direction: str, pin: int, action: str, result: str)
```
`result` is `"OK"`, `"TIMEOUT"`, or `"SENT"`. TIMEOUT is recorded **before**
`TimeoutError` is raised, so it always appears in the Part even if the cycle
fails mid-sequence.

---

## Traceability output (JSON Lines)

File: `{traceability_path}/YYYYMM/YYYYMMDD_results.jsonl`
One JSON object per line, 3 sections → 4 DB tables:

```json
{
  "part": {
    "part_id": "20260518-0042",
    "model_id": "nissan_shroud",
    "date_inspected": "2026-05-18",
    "time_inspected_s": 2.34,
    "overall_status": true,
    "piece_detected": null,
    "dry_run": false
  },
  "view_results": [
    {"view": "front_view_section_1_A", "is_ok": true, "score": 0.0021, ...}
  ],
  "trigger_events": [
    {"step_number": 1, "pin": 13, "action": "wait_for_input", "result": "OK"}
  ]
}
```

---

## CSI MUX channel switching

`RpiCsiMuxAdapter.select_channel(channel)`:
1. Skips if already on the requested channel.
2. Calls `camera.stop_stream()`.
3. Sets GPIO pins (BCM 4, 17, 18 via `OutputDevice`) from `gpio_state` in `default_values.json`.
4. Runs the I2C command from `i2c_cmd` (shell subprocess).
5. Calls `camera.start_stream()`.

GPIO pin mapping for channels A/B/C/D:

| Channel | BCM 4 | BCM 17 | BCM 18 | I2C cmd (bus 10, addr 0x0) |
|---|---|---|---|---|
| A | 0 | 0 | 1 | `0x00 0x04` |
| B | 1 | 0 | 1 | `0x00 0x05` |
| C | 0 | 1 | 0 | `0x00 0x06` |
| D | 1 | 1 | 0 | `0x00 0x07` |

The MUX constructor performs a GPIO double-init cycle (`init → close → re-init`)
to release stale handles from a previous process that did not shut down cleanly.

`RpiCsiMuxAdapter.select_channel_gpio_only(channel)` (added Sept 2026, see section 24):
applies only steps 3–4 above (GPIO + I2C), skipping `stop_stream()`/`start_stream()`.
Currently unused (dead code, kept for potential future use) — the cold-boot prewarm
script (`scripts/cold_boot_camera_prewarm.py`) was redesigned (Sept 9 2026) to avoid
`CsiCameraAdapter`/`RpiCsiMuxAdapter` entirely, using raw `gpiozero`/`os.system()`
calls instead (see section 24). Unlike `select_channel()`, it never skips the
hardware switch based on the last selected channel.

---

## Camera: dual-stream setup

`CsiCameraAdapter` runs Picamera2 in dual-stream mode:
- `main` stream: `RGB888` at capture resolution (default 4608×2592) — used by `capture_frame()`
- `lores` stream: `YUV420` at preview resolution (default 320×240) — used by `get_preview_frame_to_HTML()`

`get_preview_frame_to_HTML()`: reads lores YUV → converts to RGB → JPEG-encodes → returns bytes.
`capture_frame(view_name)`: calls `capture_request()` on `main` → `make_array("main")` → releases.
Single attempt only — on any failure the instance is force-abandoned (`picam_instance = None`)
instead of retried, same pattern as `stop_stream()`/`close_camera()`. A timed-out
`_call_picam_with_timeout()` call leaves an unkillable background thread still driving the
same Picamera2 instance, so an internal retry would race a second concurrent native call on
a non-thread-safe object — this used to happen with the old `MAX_CAPTURE_RETRIES = 3` retry
loop and is the confirmed root cause of full hardware wedges after a single slow capture.
Reinitializing before trying again is the caller's responsibility.
`restore_preview_settings()`: stop → set preview exposure → start → sleep 50 ms.
Called by `SequenceExecutor` after every inference step to restore preview exposure.

Initial controls: `AwbMode=Fluorescent` always. If `preview_time_exposure > 0`:
`AeEnable=False, ExposureTime=preview_time_exposure`. If `preview_time_exposure == 0`:
`AeEnable=True` (auto-exposure). `AfMode=Manual, LensPosition=0.0` only when
`supports_af_motor=True`. Same logic applies in `restore_preview_settings()`.

---

## Crash recovery

Both `GuiInferenceAdapter._run_loop()` and `GuiSamplesAdapter._run_loop()` share
the same crash recovery pattern:
- Catches all exceptions per cycle, prints full `traceback.format_exc()`.
- Attempts camera recovery (`close_camera()` + `initialize_camera()`) before next cycle.
- After `_MAX_CONSECUTIVE_ERRORS` (default 5) consecutive failures, stops the loop
  and logs `[CRITICAL]`. The process stays alive for status queries.

For process-level crashes, `iris.service` (Option 1 / headless deployment)
provides `Restart=on-failure`, `RestartSec=5s`. **Crash-loop protection (added
2026-08-03)**: `setup/iris.service` and `setup/iris_watchdog.service` also set
`StartLimitIntervalSec=60` / `StartLimitBurst=5` — without this,
a process that crashes immediately on every start (bad hardware, corrupted
config, port already in use) would restart forever every `RestartSec=5s`
(5s spacing never trips systemd's *default* 5-starts/10s limit). With the
explicit limit, systemd instead marks the unit `failed (Result:
start-limit-hit)` after 5 failed restarts in 60s — see the "Troubleshooting"
comment block at the bottom of `iris.service` for the `journalctl` +
`systemctl reset-failed` recovery commands.

**`setup/inspection.service` removed (2026-08-05)**: it was an orphaned unit
that ran `main.py` (the interactive CLI test loop, `input()`-driven) under
systemd — fundamentally broken, since systemd services get no TTY/stdin, plus
it still had the unexpanded `~` in `WorkingDirectory=` and a hardcoded
`User=pi`/`Group=pi` (same bug class fixed in `iris.service` on 2026-08-04).
Not referenced by `commands_first_instalation_rpi.txt`, `Iris.sh`, or
`iris_watchdog.sh` — confirmed unused before deletion. The real production
entry points remain `iris.service` (Option 1) and the kiosk autostart
(Option 2), both of which run Iris (gunicorn + `GuiInferenceAdapter`), never
`main.py` directly.

**Bug found & fixed in production (2026-08-04, Pi 192.168.45.210)**:
`iris_watchdog.service`'s `[Unit]` section previously had `Wants=iris.service`
alongside `After=iris.service graphical.target`. `Wants=` makes systemd start
`iris.service` every time `iris_watchdog.service` starts — regardless of
whether `iris.service` was explicitly `disable`d for a kiosk (Model B)
deployment. Since `iris_watchdog.service` is always enabled, this silently
started `iris.service` on every boot even on Pis meant to run kiosk-only,
breaking the Option 1/Option 2 mutual exclusion (symptom: `systemctl
is-active iris` → `active` despite `systemctl is-enabled iris` → `disabled`;
two independent things end up serving port 5000-ish confusion, though in
practice `iris.service` just wins the port race and the kiosk's own
`start_iris.sh` gunicorn — if it even runs — fails to bind). Fixed by
removing the `Wants=` line, keeping only `After=` (pure ordering, never
starts anything by itself). If a Pi was set up before this fix, after
pulling the corrected `iris_watchdog.service` re-run
`sudo systemctl daemon-reload && sudo systemctl restart iris_watchdog` and
verify with `systemctl is-active iris` (should be `inactive` on kiosk Pis).
Also fixed the same day, same root cause class (per-Pi hardcoded paths that
require manual editing, easy to miss one — this is exactly how the
`iris.service` false-active bug above got introduced): `iris.service` and
`iris_watchdog.service` now use `__IRIS_USER__`/`__IRIS_HOME__` placeholder
tokens instead of hardcoding `/home/pi/...`, substituted automatically by
`setup/install_units.sh` (`sed` + `whoami`/`$HOME`) — the only thing left to
check per-Pi is the venv name in `ExecStart` (`mezt` by default, since a Pi
could have more than one). `iris-browser.desktop`'s `Exec=` now reads
`bash -c "$HOME/Desktop/InspectionApp/setup/start_iris.sh"`, which resolves at
runtime from the logged-in user's own session — **zero editing needed**,
for any username, ever (this is what silently broke kiosk autostart on Pi
192.168.45.210: the old hardcoded `/home/pi/...` path didn't exist for user
`itmsop80op90`, and `Terminal=false` hid the failure).

**Do NOT use systemd's `%h` specifier here** ("home directory of the user
in `User=`") — it looked like a way to auto-derive `WorkingDirectory=`/
`ExecStart=`/`XAUTHORITY=` from `User=` with zero extra tooling, and was
tried first, but it did **not** resolve as documented in production
(2026-08-04, Pi 192.168.45.210, systemd 252 on Debian 12): it expanded to
`/root` instead of the configured user's home, crash-looping
`iris_watchdog.service` (`CHDIR` failure) until `StartLimitBurst` kicked in
and gave up. Verified end-to-end with `install_units.sh`'s plain `sed`
substitution instead, which has no such surprise.

**Second silent ordering-cycle bug found (2026-09-09), same file**:
`iris_watchdog.service` also had `After=iris.service graphical.target` in
`[Unit]` alongside `WantedBy=multi-user.target` in `[Install]`. A unit
`WantedBy=`d by a target while also `After=`ing a target that itself
depends on the first (`graphical.target` is ordered after and requires
`multi-user.target` by systemd's own built-in definition) creates an
ordering cycle — systemd breaks it by silently deleting one job from the
boot transaction, with the only trace being `journalctl -b | grep -i
"ordering cycle"` (no error surfaced anywhere else). Reproduces on any Pi
using the Desktop/autologin image (`graphical.target` ends up part of the
boot transaction), independent of Option 1 vs Option 2. Symptom reported:
brand-new installs where, after enabling `iris`/`iris_watchdog` and
rebooting, nothing starts — no crash, no log, just silence. Fixed by
dropping `graphical.target` from `After=` (kept `After=iris.service`):
`iris_watchdog.sh`'s only DISPLAY-dependent code path (`recover()`) never
runs until `FAILURE_THRESHOLD` (3) × `POLL_INTERVAL_S` (10s) after the unit
starts — by then any real graphical session is already up, so the ordering
was never actually load-bearing. Do NOT "fix" this by changing `[Install]
WantedBy=` to `graphical.target` instead — that would silently stop the
watchdog from ever being pulled in at all on pure headless (Option 1, no
desktop environment) installs, where `graphical.target` never becomes part
of the boot transaction.

**Why gunicorn instead of the Flask dev server (`iris.service`)**: the dev
server logs every HTTP request — including the browser topbar's 3-second
status poll — to stderr, flooding both the terminal and the
`TimestampedFileLogger` file. gunicorn's `--access-logfile /dev/null`
silences per-request entries while `--log-file -` keeps application-level
`print()`/`[INFO]`/`[WARN]` output flowing to the systemd journal.

---

## Part ID generation

`GuiInferenceAdapter._generate_part_id()` reads/writes `config/counter.json`:
```json
{"date": "20260518", "counter": 42}
```
Resets to 1 when the date changes. Output: `"20260518-0042"`.

---

## Logging

`TimestampedFileLogger(logs_path, suffix)`:
- Redirects `sys.stdout` **and** `sys.stderr` (captures all `print()` and tracebacks).
- Writes to terminal and `{logs_path}/YYYYMM/YYYYMMDD_log-{suffix}.txt` simultaneously.
- Rotates automatically at midnight without restarting the process.
- Used as a context manager in `main.py`.

Log format: `2026-05-18 14:30:22.341 - [INFO] message here`

---

## Coding conventions

- **Language**: Python 3.10+. No type: ignore, no bare except.
- **Code language (all files)**: All source code, HTML templates, JavaScript, CSS, comments, docstrings, log messages, button labels, and UI strings must be in **English**. This includes every file in `app/`, `iris/`, `scripts/`, and `config/`. The only exceptions are: (1) user-defined string *values* in configuration files (e.g. `part_model`, step descriptions set by the operator), and (2) conversation between the developer and the assistant.
- **Docstrings**: Google style (`Args:`, `Returns:`, `Raises:`, `Attributes:`, `Note:`).
- **Private members**: all prefixed with `_`. Public API = interface methods + entry points.
- **Imports**: absolute (`app.src.…`). No relative imports.
- **No logic in adapters**: adapters translate; domain logic lives in `core/`.
- `AppFactory` is the only file that imports concrete adapter classes.
- Do not add error handling for scenarios that cannot happen.
- Do not refactor beyond what is requested.
- **Convention consistency**: When adding new code, always check that variable names, patterns, and idioms match the surrounding file. If the user notices an inconsistency, fix it immediately. The user may periodically ask for a convention review across all recently modified files — treat this as a normal part of the workflow.

---

## Versioning

- **Source of truth**: `app/VERSION` — a single-line semver string (`MAJOR.MINOR.PATCH`, e.g. `1.1.0`), read at startup by `IrisServer.py` (`_read_iris_version()`) and exposed as `iris_version` in `GET /api/info`. Before 2026-08-03 this endpoint returned a hardcoded, never-updated `"1.0.0"` — now it reflects `app/VERSION` for real.
- **Convention (semver, adopted 2026-08-03)**: pick `v1.0.0` as the baseline for "pre-2026-07-31 features" (before the big batch of Iris features documented in "Pending work" below). From there:
  - **PATCH** (`1.0.x`): bug fixes only, no new user-facing behavior.
  - **MINOR** (`1.x.0`): new features / additive endpoints, no breaking changes to existing behavior or API shapes. This covers almost everything in the "Pending work" table below (clone sequence, review/relabel gallery, NOK-only persistence, Schedule Timed Captures, Restart Iris, View Production NOK viewer, etc.) — hence the jump straight to **`1.1.0`** on 2026-08-03 to cover the entire 2026-07-31 + 2026-08-03 batch in one bump (no version existed in between, so no need for several intermediate minors).
  - **MAJOR** (`x.0.0`): reserved for breaking changes (e.g. an incompatible sequence JSON schema change, or removing/renaming an existing API endpoint's request/response shape).
- **When shipping a new feature**: bump `app/VERSION` as part of the change (MINOR for additive features, PATCH for pure bug fixes) and mention the new version in the corresponding "Pending work" row below.

---

## Technology stack

| Layer | Library |
|---|---|
| Camera | `picamera2`, `libcamera` |
| GPIO | `gpiozero` |
| Inference | `tflite_runtime`, EdgeTPU delegate (`libedgetpu.so.1`) |
| Image processing | `numpy`, `opencv-python` (`cv2`), optional `scipy` |
| Logging | stdlib only (`sys`, `os`, `datetime`) |
| Concurrency | `threading` (inspection loop in daemon thread) |
| Service management | `systemd` (see `setup/iris.service`, `setup/iris_watchdog.service`) |
| Web server (dev) | Flask built-in dev server (`python3 -m iris.IrisServer`) |
| Web server (prod) | `gunicorn` with `gthread` worker — see `gunicorn.conf.py` and `setup/iris.service` |

---

## Backend status & key implementation gotchas

The full hexagonal implementation (all interfaces/adapters/core services listed in
the Architecture section above), multi-platform support (PC/RaspberryPi/Jetson),
and the PaDiM calibration pipeline (see "PaDiM calibration" section below) are
complete. The notes below are **non-obvious facts and past bug fixes** worth
knowing before touching related code — most are not written down anywhere else.

- `SequenceExecutor.get_last_captured_frames() -> dict[str, np.ndarray]`: thread-safe getter for frames from the most recent cycle (`_last_captured_frames` + `_frames_lock`). `GuiInferenceAdapter.get_last_inference_frames()` delegates to it.
- `GuiInferenceAdapter(images_path: str | None)`: when set, calls `_save_latest_frames()` after each successful cycle to persist JPEG thumbnails (≤ 800 px wide) to `images_path/latest/{view_name}.jpg` via atomic `os.replace()`. `IrisServer.api_frame()` uses this as a disk fallback (survives server restarts) when `IrisState` has no manual preview for the channel.
- **`IrisState.is_busy` vs `is_running`**: `controller_stopping: bool` is set `True` by `api_stop` before `ctrl.stop_loop()` and cleared after the join completes. `is_busy` returns `is_running or controller_stopping`. Routes that precede a hardware reinit (`_load_sequence`, `initialize_hardware`, `factory.shutdown`) must guard with `is_busy` — NOT `is_running` — to avoid touching the camera while the background thread is still tearing down. Routes that only guard against double-start/config changes use `is_running`.
- `SequenceBuilderService.find_sequence_by_part_model()` (duplicate detection); `list_sequences()` excludes `sequence_draft.json`.
- `CpuInferenceAdapter`/`CoralUsbInferenceAdapter.load_models_from_directory()` emits `[WARN]` (not raise) when the models directory is empty — normal before first training. `CpuInferenceAdapter` imports `tflite_runtime` first, falls back to `tensorflow.lite.Interpreter` on PC.
- CORS is enabled (`flask-cors`) for Abigail cross-origin requests, currently `origins: "*"` — restrict to the Abigail IP in production.
- `gunicorn.conf.py` `timeout=0` (infinite) is intentional — with `worker_class=gthread`, gunicorn only monitors the process heartbeat, not individual threads, so a non-zero timeout would kill the worker erroneously during a long calibration sweep. **Do NOT change it.**
- **`CsiCameraAdapter` libcamera stall protection**: `_call_picam_with_timeout(fn, *args)` runs any blocking Picamera2 call in a daemon thread, raises `RuntimeError` after `_PICAM_CALL_TIMEOUT_S = 1.0 s` (lowered from `8.0 s` — see section 23). Protects `capture_request()`, `capture_array()`, and critically `stop()`/`close()` inside `close_camera()` (a stalled ISP blocks these too — unprotected, they'd hang the crash-recovery thread forever). On timeout, the instance is force-abandoned (`picam_instance = None`, `_is_initialized = False`) so `initialize_camera()` always starts clean — `close_camera()`'s own `stop()`/`close()` except blocks now do this too (previously only `stop_stream()`/`start_stream()` did, see section 23).
- **`GuiInferenceAdapter._run_loop()` recovery path always calls `self.resume()` in a `finally` block** — otherwise a failed recovery leaves the loop paused forever instead of letting `_consecutive_errors` keep incrementing toward `_MAX_CONSECUTIVE_ERRORS` (clean exit + systemd restart). See "Camera reconnection robustness" row in Pending work for the `_USE_FULL_HARDWARE_RECOVERY` rollback flag around this same call site.
- **`IrisServer._camera_keepalive`**: daemon thread reading one preview frame every 5 min while all loops are idle, to prevent the libcamera ISP pipeline from stalling during extended idle periods. Skipped during calibration tasks (would interfere with MUX switching).
- **`state.lock` must never be held across `initialize_hardware()`**: `_on_calibration_done`/`_on_done` used to call `_load_sequence()` (→ `initialize_hardware()`) inside `with state.lock:`. If libcamera stalled during reinit, the lock stayed held indefinitely, freezing every other request (status poll included). Fix: read `seq_path`/`factory_ref` inside the lock, release it, do the reinit outside the lock, then re-acquire only to commit the resulting state.
- `IrisServer._on_sweep_done` persists `{model_path}/sweep_results.json` (`best_blocks` + full `sweep_results`) so Step 2/3 survive a restart without repeating the sweep. `api_calibration_sweep_results` and `api_calibration_status` auto-load from it (priority: `eval_config.json` → `sweep_results.json` → empty) when RAM state is empty post-restart.
- `canvas_tools.js CanvasTools.prototype.resize(w, h)`: resizes the Fabric.js canvas to actual card dimensions, re-scales background + shapes from capture-resolution coords, and saves/restores the detection ROI around `setDimensions()` (previously lost on resize). `exportPipeline()`/`exportDetectionRoi()` use `getScaledWidth()/getScaledHeight()` (not `getBoundingRect()`) to avoid a ~2px stroke-bias growth on every "Apply to pipeline" click.
- `builder.js`: `resizeAllCanvases()` runs in `requestAnimationFrame` before restoring the draft so canvases fill their cards from the start; a `ResizeObserver` keeps them matched on every later layout change.
- `iris.css .camera-section`: `display:flex; flex-direction:column` with `.camera-grid { flex:1; min-height:0 }` so the inspection camera grid fills available height responsively.
- `SequenceSettings._TOOL_ORDER`: canonical pipeline execution order regardless of JSON order — `apply_clahe(0)`, `put_black_circle/put_black_rectangle(1)`, `apply_roi_crop(2)`, `resize_to_training_resolution(3)`, `normalize_mobilenet(4)`.
- `CalibrationService._apply_masks()` applies `put_black_circle`/`put_black_rectangle` to the full-resolution image before crop; `IrisServer._build_view_configs()` extracts mask tools from the pipeline into `view_cfg["masks"]`.
- **`_build_view_configs()` returns `list[dict]`, not `dict`** — never call `.items()` on it. To recalibrate a subset of views, pass a filtered `view_names` list.
- **`TraceabilityReviewService`** (`app/src/core/services/TraceabilityReviewService.py`): pure core service. Reads JSONL traceability files, computes per-view NOK stats, detects drift patterns, and moves (not copies, since 2026-08-03 — see "View Production NOK" row below) inference images to training dirs. Injected via `AppFactory.create_traceability_review_service()`.
  - `analyze(date_str) → ReviewAnalysis`: reads daily JSONL, classifies drift as `"global"` (≥60% of views NOK in same cycle) or `"isolated"` (single-view pattern). Counts available images via timestamp matching. Sets `total_nok_parts` = number of parts where at least one view classified NOK (used for global %NOK in modal header).
  - `_IMAGE_MATCH_WINDOW_S = 5`: margin in seconds around the full cycle window when matching image filenames. Match window is `[date_inspected − 5s, date_inspected + duration_s + 5s]`. Images may be saved at capture time (start of cycle) or after inference (end of cycle); this range covers both cases without false matches (inter-cycle gap ≥15 s).
  - `list_nok_images(date_str, view_name, page=1, page_size=20) → (items, total)`: paginates NOK images for a view/date. Each item: `{filename, abs_path, score, time_str}`. `time_str` = `HH:MM:SS` from filename.
  - `promote_images(date_str, view_names, test_count=5, train_count=20, excluded_filenames=None) → dict`: hard-example mining — `train/OK` gets the `train_count` images with the **worst anomaly score** (highest score = farthest from Gaussian centroid) with **train priority**: when few images are available they all go to train rather than test. `test/OK` gets the `test_count` most-recent images from the remainder after train. Images whose basename is in `excluded_filenames` are silently skipped before the split. Remaining (non-selected) images are simply left in place in `inference_images_path`, still browsable/deletable via `list_all_nok_images_for_view()`. **`_move_images()` (renamed from `_copy_images()`, 2026-08-03)**: uses `shutil.move` instead of copy — promoted images no longer stay duplicated in `inference_images_path`; if the destination filename already exists (re-promote edge case), the source is deleted instead of overwritten.
  - `list_all_nok_images_for_view(view_name, page=1, page_size=20) → (items, total)` (2026-08-03): same item shape as `list_nok_images()` plus `date_str`, but scans **every date** in the traceability history instead of one — backs the Step 1 "View Production NOK" viewer described below. Sorted newest-first (vs. worst-score-first for promote's hard-example mining). Skips building the (filesystem-listing) image index for any day with zero NOK records for the requested view, to keep the full-history scan cheap.
  - `delete_nok_images(paths) → {"deleted": int, "errors": [...]}` (2026-08-03): permanently deletes inference images, validated to resolve inside `inference_images_path` (mirrors how `CaptureReviewService.delete_images()` guards `images_path`).
- **`TraceabilityReviewModels`** (`app/src/core/models/TraceabilityReviewModels.py`): `ViewReviewStats` and `ReviewAnalysis` dataclasses. `ReviewAnalysis.total_nok_parts: int` = parts where at least one view was NOK (used for global NOK rate in modal header).
- **Feature: Recalibrate from Production** (Step 3, calibration page):
  - Button: "Recalibrate from Production…" → opens `#review-modal-overlay`
  - Modal: date picker → Analyze → **summary line** shows `N parts inspected on DATE · M NOK (X.X%)` where M/X.X% is the global per-part NOK rate (a part is NOK when at least one view is NOK); per-view table (NOK%, score range, image button/count, drift badge) → operator selects views → two action buttons in footer: **Promote only** (`#btn-review-promote-only`) and **Promote & Recalibrate** (`#btn-review-promote`). Both are enabled/disabled together by `syncPromoteButton` whenever at least one checkbox is ticked.
  - "Images" column: shows `🔍 N NOK` button when images exist; clicking calls `openImageGallery(title, dateStr, viewName)` which opens `#gallery-modal-overlay`
  - Image gallery: paginated grid of NOK thumbnails (`/api/image_file?path=<abs_path>`), per-card exclude toggle (bidirectional), lightbox on click
  - **Exclude persistence**: `_reviewExcluded: Map<viewName, Set<filename>>` (module-level, reset in `openReviewModal`). When a gallery opens, its `excluded` Set is pre-populated from `_reviewExcluded.get(viewName)` — so already-excluded images appear visually marked on re-open. When a gallery closes, `_reviewExcluded.set(viewName, new Set(excluded))` writes back the full current Set (supports un-exclude). `_getAllExcluded()` helper flattens all view Sets plus the currently-open gallery into a single deduplicated array for the promote request.
  - Drift badges: `global` (yellow, ≥60% views fail same cycle), `isolated` (orange, single-view pattern), `none` (blue)
  - Backend: `GET /api/calibration/review_analysis?date=YYYYMMDD` (returns `total_nok_parts`), `GET /api/calibration/review_images?date=YYYYMMDD&view_name=…&page=…&page_size=…`, `GET /api/image_file?path=<abs_path>` (security: path must start with `realpath("data/") + os.sep`), `POST /api/calibration/promote_and_recalibrate` `{date_str, view_names, test_count, train_count, excluded_images, params}`, `POST /api/calibration/promote_only` `{date_str, view_names, test_count, train_count, excluded_images}` — copies images to train/OK and test/OK without starting calibration; does not require the inspection loop to be stopped.
  - **Export TSV**: `GET /api/calibration/export_traceability?date=YYYYMMDD` → downloads `traceability_{model_id}_{date}.txt` (UTF-8, tab-separated). Columns are discovered **dynamically** from all records in the JSONL: fixed base cols (`part_id`, `model_id`, `date_inspected`, `duration_s`, `overall_status`, `piece_detected`), then one group of 4 columns per GPIO step (`step{N}_direction`, `step{N}_pin`, `step{N}_action`, `step{N}_result`, ordered by step number), then one group of 4 columns per view (`{view_name}_classification`, `{view_name}_score`, `{view_name}_threshold_min`, `{view_name}_threshold_max`, ordered by first appearance). Fully scalable: works with any number of views or GPIO events. Triggered by `⬇ Export TSV` button (`#btn-export-traceability`) in the review modal footer via `window.location.href`; button is disabled until Analyze succeeds.
  - JS functions: `openReviewModal`, `closeReviewModal`, `runReviewAnalysis`, `renderReviewAnalysis`, `syncPromoteButton`, `runPromoteAndRecalibrate`, `runPromoteOnly`, `openImageGallery`, `loadGalleryPage`, `renderGalleryPage`, `openLightbox`, `closeLightbox`, `updateExcludedCount`, `_getAllExcluded`
  - Gallery state: `_galleryState` module-level object; `excluded: Set<string>` (basenames) pre-populated from `_reviewExcluded` on open, written back on close. `_reviewExcluded: Map<viewName, Set<string>>` persists across gallery open/close within the same review session.
  - CSS classes: `.modal-box`, `.modal-box-wide`, `.review-drift-banner`, `.drift-badge`, `.gallery-grid`, `.gallery-card`, `.gallery-card.excluded`, `.gallery-card-meta`, `.gallery-card-score`, `.gallery-card-exclude`, `.gallery-pagination`, `.lightbox-overlay` (`z-index: 3000` — must be above `.modal-overlay` at 2000), `.lightbox-content`, `.lightbox-nav`, `.lightbox-meta`, `.lightbox-close`
  - **Future — excluded → NOK promotion**: add `send_excluded_to_nok: bool = False` param to `promote_images()`; locate `abs_path` for each excluded filename via `_build_image_index()`; copy to `images_path/test/NOK/{channel}/`. Requires a dedicated viewer (same gallery component) to review the images before they are used as NOK training data. Not yet implemented.
- **Feature: View Production NOK images** (Step 1, calibration page) — ~~Implemented 2026-08-03~~: new "View Production NOK…" button next to the Step 1 review controls (`.cal-review-controls`), opens `#nok-viewer-modal-overlay`. Unlike "Recalibrate from Production" (which is date-scoped and promote-oriented), this viewer has **no date picker** — it loads every NOK inference image on record for the selected view, across the entire traceability history, via `GET /api/calibration/nok_images_all?view_name=…&page=…&page_size=…` (backed by the new `list_all_nok_images_for_view()`, sorted newest-first). Its purpose is disk-space cleanup: a selection bar + checkboxes (reusing the exact same `.review-card-checkbox`/`.review-selection-bar` CSS as Step 1's own gallery) plus a lightbox (Delete-only action, no relabel/move-to-set buttons — this view isn't part of the calibration image tree) let the operator permanently delete NOK inference images no longer needed, via `POST /api/calibration/delete_nok_images {paths}` → `TraceabilityReviewService.delete_nok_images()` (validates every path resolves inside `inference_images_path`, mirrors `CaptureReviewService.delete_images()`'s guard on `images_path`). This viewer is also why `promote_images()` was switched from copy to move (row above): once an image is promoted it's still browsable, just now under Step 1's own "Train OK"/"Test OK" tabs instead of `inference_images_path` — so keeping a duplicate around no longer serves any purpose. Validated with `py_compile` only (no local venv with pytest in this environment).
- **Feature: Current model evaluation panel** (Step 3, calibration page):
  - Shows AUC, sep_ratio, block and thresholds for the **active fitted model** (distinct from sweep Step 2 metrics)
  - Auto-loads on page load from `calibration_eval.json`; auto-refreshes when any calibration completes (polls `/api/calibration/progress` until `running=false`)
  - ↻ Refresh button for manual reload
  - Color coding: green = sep≥2.0 / AUC≥0.90; yellow = sep≥1.2 / AUC≥0.70; red = below
  - Backend: `GET /api/calibration/calibration_eval`; helper `_write_calibration_eval(results, model_path)` called from both `_on_calibration_done` and `_on_done` callbacks
  - JS functions: `loadCalibrationEval`, `renderCalibrationEval`, `waitForCalibrationDone`
  - CSS classes: `.cal-eval-block`, `.cal-eval-table`, `.cal-eval-header`, `.eval-good`, `.eval-warn`, `.eval-bad`

## PaDiM training workflow (for views with an existing large dataset)

When a view already has a curated train/test dataset (≥ 200 train OK, ≥ 50 test OK,
≥ 3 test NOK), follow this iterative workflow to calibrate a PaDiM model:

### Step 1 — Inventory the dataset
```bash
find 00-Imagenes/entrenamiento/{VIEW}/OK -type f | wc -l   # train OK
find 00-Imagenes/test/{VIEW}/OK           -type f | wc -l  # test OK
find 00-Imagenes/test/{VIEW}/NOK          -type f | wc -l  # test NOK
```
Check that all capture dates present in the test set are also represented in the
training set. A **bimodal OK score distribution** (visible in the histogram) is the
clearest symptom that a capture date in the test set is NOT in training. The fix is
always to add a small sample of that date to training — never remove from test.

### Step 2 — Configure training_padim.py
Edit these constants for the target view:
```python
VIEW           = "primera_mitad/D"    # path under entrenamiento/ and test/
VIEW_NAME      = "primera_mitad_D"    # used for file names
Y0, X0, H, W  = 170, 200, 550, 750   # ROI crop (height, width)
TRAINING_SHAPE = (550, 750)           # must match H, W
USE_CLAHE      = False                # default; only experiment if Sep < 1.1x
```
ROI reference (primera_mitad / segunda_mitad views):
| View | Y0 | X0 | H | W |
|---|---|---|---|---|
| A | 135 | 400 | 525 | 525 |
| B | 150 | 375 | 525 | 525 |
| C | 135 | 280 | 525 | 525 |
| D | 170 | 200 | 550 | 750 |

### Step 3 — Run the full single-block sweep
```bash
cd /home/alan-jafet/ML/Frambuesa/src/02-Pytorch
source /home/alan-jafet/venvs/visredPC/bin/activate
python3 sweep_padim.py --mode full 2>&1 | tee sweep_{VIEW_NAME}_singles_$(date +%Y%m%d_%H%M%S).log
```
`sweep_padim.py` reads VIEW/VIEW_NAME/ROI directly from `training_padim.py` imports.
`--mode full` tests all 15 single blocks (b1–b15). Do NOT add `--combos` initially.
Results are auto-saved to `sweep_results_{VIEW_NAME}.txt` and best config to
`best_block_config_{VIEW_NAME}.json`.

### Step 4 — Analyze sweep results
Look for:
- **AUC** = 1.0000 on multiple blocks → healthy separation
- **Sep ratio** (min(NOK) / max(OK)) — target ≥ 1.25x
- **Bimodal OK** in individual block histograms → dataset date-coverage gap

If Sep < 1.1x on ALL blocks:
1. List capture dates in test set vs. training set and find the missing date.
2. Add 30–60 images from that date to training.
3. Re-run sweep from step 3.

### Step 5 — Final calibration
Once sweep gives Sep ≥ 1.2x, run the full calibration to produce deployable files:
```bash
python3 training_padim.py 2>&1 | tee resultados_{VIEW_NAME}_final_$(date +%Y%m%d_%H%M%S).log
```
Output dir: `nissan/{VIEW_NAME}/{timestamp}/`
Key output files:
- `padim_{VIEW_NAME}_params.npz` — mean, precision, random_idx
- `thresholds.json` — per-view {min, max}
- `eval_config.json` — top_k_pixels, border_crop_px, gaussian_sigma
- `teacher_mobilenetv2_backbone.onnx` — backbone copy for deployment

### Step 6 — Validate results
Expected in the score distribution graph:
- OK cluster is compact (std < 0.6 ideally), no bimodal shape
- NOK cluster clearly separated, gap > 1.5 absolute points
- Sep ratio ≥ 1.25x (warn if < 1.2x after multiple iterations)

If CLAHE was tried and Sep dropped → revert immediately (`USE_CLAHE = False`).
CLAHE tends to make NOK features look more "normal", reducing min(NOK).

### Key constants (scoring — shared across all views)
```python
K_PIXELS    = 2      # top-k pixel mean for anomaly score
BORDER_CROP = 1      # pixels cropped from error map border
SIGMA       = 0.0    # Gaussian blur (0 = off)
PADIM_DIM   = 100    # PaDiM random projection dimension
PADIM_LAMBDA= 0.01   # regularization for precision matrix
```

### Threshold formulas (generated automatically by training_padim.py)
```
max_threshold = percentile_99.5(OK_scores) * 1.10   [robust to bimodal OK]
min_threshold = min(OK_scores) * 0.90
```

---

## PaDiM calibration (all platforms)

The calibration pipeline runs on **any supported device** — PC, Jetson, and Raspberry Pi.
All dependencies (`onnxruntime`, `scikit-learn`, `scipy`, `numpy`, `opencv-python`) are
pure Python and available on every platform. `AppFactory` selects the appropriate camera
and GPIO adapters automatically based on `device_type`, so the calibration code itself
has no platform-specific branches.

**Platform differences handled by AppFactory:**

| Platform | Camera adapter | GPIO adapter | GPIO trigger |
|---|---|---|---|
| RaspberryPi | `CsiCameraAdapter` | `RpiGpioAdapter` | Physical PLC signal |
| Jetson | `CsiCameraAdapter` or `UsbCameraAdapter` | `RpiGpioAdapter` | Physical signal |
| PC | `UsbCameraAdapter` | `NullGpioAdapter` | Immediate (no wait) |

On PC, `NullGpioAdapter.wait_for_input()` returns immediately (no blocking), so the
capture loop fires on each iteration without a physical trigger. This is intentional
for desktop development and dataset building without hardware.

### Backbone ONNX pre-installation (one-time system setup)

MobileNetV2 feature extraction uses pre-exported ONNX backbone files. These are
**platform-agnostic** — the same files work on PC (development) and RPi (on-device
calibration). They are shared across all products and views.

**Storage location**: `data/models/backbones/teacher_mobilenetv2_backbone_b{N}.onnx`
**Range**: b3 through b17 (15 files, ~15 MB each, ~230 MB total)
**Install once** per device before first calibration. On RPi and Jetson (no internet
access after setup), copy the files manually. On PC they can be exported directly
with the included export script.

**Resolution independence**: All backbone ONNX files must have dynamic spatial axes
(`{0: "batch", 1: "height", 2: "width"}` for both input and output) to support any
`resize_to_training_resolution` value. If a backbone was generated without dynamic
axes, inference raises `INVALID_ARGUMENT: Got: <H> Expected: 525`. Fix with `--force`.

MobileNetV2 block guide:
- b1–b2: low-level edge/color features — not useful for anomaly detection
- b3–b14: progressively richer semantic features — main sweep range
- b15–b17: high-level abstract representations — worth testing in difficult views
- b18+: classifier layers — not useful

#### Step 1 — Export backbone ONNX files (PC only, once)

Use the standalone script included with InspectionApp. It does **not** depend on
Frambuesa — only `torch` and `torchvision` are needed (already in the development
virtualenv).

```bash
cd /home/alan-jafet/ML/InspectionApp
source /home/alan-jafet/venvs/visredPC/bin/activate

# Export all 15 backbones (b3–b17) with dynamic spatial axes:
python3 scripts/export_backbones.py

# Export specific blocks only:
python3 scripts/export_backbones.py --blocks 9 13

# Re-export (overwrite) and verify at a given resolution:
python3 scripts/export_backbones.py --force --verify --verify-shape 525 525
```

Flags:
- `--blocks N …` — specific block numbers (default: 3–17)
- `--output DIR` — destination directory (default: `data/models/backbones/`)
- `--force` — overwrite existing files
- `--verify` — run onnxruntime inference sanity check after each export
- `--verify-shape H W` — spatial size to use for the verify step (default: 224 224)

Output: `teacher_mobilenetv2_backbone_b3.onnx` … `teacher_mobilenetv2_backbone_b17.onnx`
in `data/models/backbones/`.

#### Step 2 — Copy backbones to RPi / Jetson

```bash
# From PC, push to device (replace PI_IP):
scp /home/alan-jafet/ML/InspectionApp/data/models/backbones/teacher_mobilenetv2_backbone_b*.onnx \
    pi@<PI_IP>:/home/pi/InspectionApp/data/models/backbones/
```

#### Alternative — offload the entire sweep to a PC (`scripts/offload_calibration.py`)

Runs the full sweep + calibration fit on a PC/laptop (~15–20 min on modern CPU vs ~8 h on the Pi) and pushes the resulting model files back to the Pi.

**PC requirements:** `pip install onnxruntime numpy scipy scikit-learn opencv-python Pillow matplotlib`

**Minimum files needed on the PC** (no Flask/IrisServer required):
- `app/src/core/services/CalibrationService.py`
- `app/src/core/models/CalibrationModels.py`
- `app/src/adapters/output/PaDiMFeatureExtractorAdapter.py`
- `app/src/interfaces/IPaDiMFeatureExtractor.py`
- `scripts/offload_calibration.py`

**Usage** (run from the directory that contains `app/` and `scripts/`):
```bash
# Basic — fetches everything from the Pi automatically (default Pi path: /home/testing/Desktop/InspectionApp)
python3 scripts/offload_calibration.py --pi 192.168.45.206

# Restrict sweep to a block range (faster when you already know the useful range)
python3 scripts/offload_calibration.py --pi 192.168.45.206 --blocks 7 13

# Sweep only — inspect the table before committing to the fit
python3 scripts/offload_calibration.py --pi 192.168.45.206 --sweep-only

# Pi with project in a non-default path
python3 scripts/offload_calibration.py --pi 192.168.45.207 --pi-path /home/testing/InspectionApp

# Custom label (default label = Pi IP — used as folder name under data/offload/)
python3 scripts/offload_calibration.py --pi 192.168.45.206 --label linea_A_puesto_1
```

**Key behaviour:**
- The sequence JSON is always fetched via `scp` from the Pi at startup — no local `config/` folder needed. Multiple Pi's with different sequences never collide.
- Local layout: `data/offload/<label>/images/<part>/` (images) + `data/offload/<label>/models/<part>/` (results). Backbone ONNX files are shared at `data/models/backbones/`.
- After fit, results (`padim_*_params.npz`, `thresholds.json`, `eval_config.json`, graphs) are rsync'd back to `<pi-path>/data/models/<part>/` on the Pi.
- `--no-pull` / `--no-push` skip the rsync steps.
- The default `--pi-path` is `/home/testing/Desktop/InspectionApp` — matching the standard Pi installation location.



On every device that will run calibration (PC, RPi, Jetson):

```bash
cd /path/to/InspectionApp
pip install -r requirements.txt
# or individually, if only adding calibration:
pip install onnxruntime scikit-learn scipy
```

`onnxruntime`, `scikit-learn`, and `scipy` are all in `requirements.txt`.
`onnxruntime` is available for x86_64, aarch64 (RPi 64-bit, Jetson) via PyPI.
No `torch` or `tensorflow` is required on the device for calibration or inference.

Backbone files are generated once on PC (`Frambuesa/src/02-Pytorch/`) and copied to
`data/models/backbones/` on each device.

### IPaDiMFeatureExtractor + PaDiMFeatureExtractorAdapter

`app/src/interfaces/IPaDiMFeatureExtractor.py`:
```python
class IPaDiMFeatureExtractor:
    def load_backbone(self, backbone_path: str) -> None: ...
    def extract_features(self, image_rgb: np.ndarray) -> np.ndarray: ...
    # image_rgb: (H, W, 3) uint8 — returns (H', W', C') float32 feature map
```

`app/src/adapters/output/PaDiMFeatureExtractorAdapter.py`:
- Loads backbone with `onnxruntime.InferenceSession`
- Preprocesses: optional CLAHE, ROI crop, resize to training shape, MobileNetV2 normalization
- Returns raw feature map as numpy array (no Gaussian fitting — that is done by `CalibrationService`)

### CalibrationService

`app/src/core/services/CalibrationService.py` — pure core service, no adapter imports.

Dependencies (injected via constructor):
- `IPaDiMFeatureExtractor` — for backbone ONNX feature extraction

Responsibilities:
- `run_sweep(view_configs, image_dirs, backbones_dir, blocks, params, progress_cb)` → `SweepResult`
  - Iterates blocks b3–b17 (default; configurable per call)
  - For each block: load backbone → extract features from train/OK images → fit Gaussian →
    score test/OK + test/NOK → compute AUC + Sep ratio + OK-distribution stats (`min_ok`,
    `mean_ok`, `std_ok`, `cv_ok`)
  - Reports progress via `progress_cb(step, total, message)` for Iris live updates
  - Does **not** write files — returns `SweepResult` to caller
- `run_calibration(view_configs, image_dirs, backbones_dir, block, params)` → `CalibrationResult`
  - Fits final Gaussian model for the selected block
  - Writes all output files to `model_path`

`SweepResult` and `CalibrationResult` are plain dataclasses in `core/models/`.

**Scoring params** accepted by both methods (all optional with defaults):
```python
top_k_pixels:   int   = 2      # top-k pixel mean for anomaly score
border_crop_px: int   = 1      # pixels cropped from error map border
gaussian_sigma: float = 0.0    # Gaussian blur sigma (0 = off)
use_clahe:      bool  = False  # apply CLAHE before feature extraction
padim_dim:      int   = 100    # PaDiM random projection dimension
padim_lambda:   float = 0.01   # regularization for precision matrix
min_sep_gate:   float = 1.0    # run_sweep only — see "Best-block selection rule" below
```

**Best-block selection rule (`CalibrationService._build_sweep_result`)**:

Thresholds are derived purely from the OK score distribution (see `eval_config.json`
section above), so between-distribution separation (`sep_ratio = min_nok / max_ok`) can be
noisy when `n_test_nok` is small (a handful of NOK samples per view is common). To make the
automatic block choice more robust, `BlockSweepResult` also reports the **coefficient of
variation of the OK scores** (`cv_ok = std_ok / mean_ok`) — a measure of how tight/consistent
the OK distribution is for that block, independent of NOK availability.

Selection logic per view:
1. `candidates = [b for b in block_results if b.sep_ratio > min_sep_gate]` (default gate `1.0`).
2. If `candidates` is non-empty: `best = min(candidates, key=lambda b: b.cv_ok)` — the block
   with the most consistent OK distribution among those that already clear the safety gate.
3. If no block clears the gate (e.g. too few/no NOK samples, or a genuinely hard view):
   `best = max(block_results, key=lambda b: b.sep_ratio)` — falls back to the historical
   "highest separation wins" rule, so behavior is unchanged for gate-less/legacy configs.

`min_sep_gate` is configurable per call via `params["min_sep_gate"]` (UI: "Min. separation
gate" field in Step 2 of `calibration.html`, CLI: `--min-sep-gate` in `offload_calibration.py`).
`SweepResult.best_cv_ok` reports the CV of the winning block.

**⚠ Keep in sync**: `CalibrationModels.py`, `CalibrationService.py`, and
`scripts/offload_calibration.py` must be identical (aside from import paths) between this
repo and the offline PC toolkit at `calibrate_from_pc/` (see "Alternative — offload the
entire sweep to a PC" below). `iris/IrisServer.py`, `calibration.html`, and `calibration.js`
are Pi-only (no Flask UI in the offline toolkit) — changes to sweep/calibration *logic*
must be mirrored in both places; changes to the *Iris web UI* only need to happen here.

### GuiCalibrationAdapter

`app/src/adapters/input/GuiCalibrationAdapter.py` — thin input adapter.

- Receives `CalibrationService` (one service, following invariant 4)
- Runs sweep and calibration in background threads; never blocks the Flask thread
- Reports progress to `IrisState.calibration_progress`
- Crash recovery: catches exceptions per task, logs full traceback

### Calibration Capture mode

Shared with "Samples" mode — both write to the same unified image tree.
Since PaDiM is one-class (trains only on OK), the label routing is:

| Mode | Label selected | Destination |
|---|---|---|
| Samples | OK | `images_path/train/OK/{channel}/` |
| Samples | NOK | `images_path/test/NOK/{channel}/` |
| Calibration capture | Train OK | `images_path/train/OK/{channel}/` |
| Calibration capture | Test OK | `images_path/test/OK/{channel}/` |
| Calibration capture | Test NOK | `images_path/test/NOK/{channel}/` |

**Unified image directory structure** (`images_path = ./data/images/{part_model}/`):
```
data/images/{part_model}/
    train/OK/{channel}/   ← Samples OK  +  Calibration train_ok
    test/OK/{channel}/    ← Calibration test_ok
    test/NOK/{channel}/   ← Samples NOK  +  Calibration test_nok
    inference/YYYYMM/YYYYMMDD/ ← Inference saved images (written by LocalStorageAdapter)
                             Files: YYYYMMDDHHmmss_{view_name}.jpg — one per view per cycle,
    latest/               ← Most recent inference frame per view (written atomically by
                             GuiInferenceAdapter after each cycle; survives restarts)
```

Capture is GPIO-triggered (same mechanism as Samples mode) on RPi and Jetson.
On PC, `NullGpioAdapter` makes the trigger fire immediately on each iteration.
The user selects the target set (Train OK / Test OK / Test NOK) in Iris before
starting the capture loop.

**Sequence JSON path** (auto-generated by `/api/setup/finalize`):
```json
"images_path": "./data/images/{part_model}/"
```

`AppFactory.create_samples_controller()` reads `images_path` with fallback to
legacy `train_images_path` for backward compatibility.

### Output files written by CalibrationService

All written to `model_path` (`./data/models/{part_model}/`):
```
sweep_results.json              ← persisted by IrisServer._on_sweep_done after every sweep
                                   (best_blocks dict + full block_results per view)
                                   Auto-loaded on server restart by api_calibration_status
                                   and api_calibration_sweep_results so Step 2 and Step 3
                                   survive a page reload without repeating the sweep.
padim_{view_name}_params.npz    ← Gaussian params (mean, precision, random_idx)
thresholds.json                 ← per-view {min, max}
eval_config.json                ← scoring params used (includes use_clahe + block + sep_ratio)
graphs/
    {view_name}_score_dist_{timestamp}.png            ← OK/NOK score histogram with threshold lines + metrics box
    {view_name}_heatmap_{ok|nok}_{N}_{timestamp}.png  ← 3-panel: original / error map / overlay
                                                        first 5 OK + ALL NOK from test set
```

Error map colour scale is normalised globally across all heatmap entries for that view so OK and NOK maps are directly comparable.

**matplotlib compatibility note**: `_save_calibration_graphs()` uses `matplotlib.colormaps.get_cmap("jet")` — NOT `cm.get_cmap()`. `cm.get_cmap` was removed in matplotlib 3.9. Never use `matplotlib.cm.get_cmap` in any new code; always use `matplotlib.colormaps.get_cmap` or `matplotlib.colormaps["name"]`.

`_extract_all_with_images()` returns `(features, raw_uint8_images)` — raw images are captured before ImageNet normalisation and are only used for display.
`_score_with_error_maps()` returns `(scores, error_maps_before_border_crop)` — full spatial error map is preserved for overlay; scoring still uses the border-cropped version.

`eval_config.json` written by `CalibrationService` (superset of the legacy format):
```json
{
    "top_k_pixels": 2,
    "border_crop_px": 1,
    "gaussian_sigma": 0.0,
    "use_clahe": false,
    "blocks": {"front_view_section_1_A": 9},
    "backbone_paths": {"front_view_section_1_A": "../backbones/teacher_mobilenetv2_backbone_b9.onnx"},
    "sep_ratio": 1.495,
    "calibration_date": "2026-06-01"
}
```
`blocks` and `backbone_paths` are dicts keyed by `view_name` (one entry per view).
`backbone_paths` values are relative to `model_path` — `PaDiMInferenceAdapter` resolves
them with `os.path.normpath(os.path.join(models_dir, backbone_rel))`.
The legacy `training_padim.py` on PC does not write them. `SequenceSettings` ignores
unknown fields (backward-compatible).

### Model file naming — disambiguation

Three adapter types load files with `teacher` in the name. They are entirely different
architectures and must never be confused:

| File pattern | Used by | Notes |
|---|---|---|
| `padim_*_params.npz` + `eval_config.json` | `PaDiMInferenceAdapter` (inference) | Gaussian params per view; backbone path read from `eval_config["backbone_paths"]` |
| `teacher_{view_name}_float32.tflite` | `CoralUsbInferenceAdapter` / `CpuInferenceAdapter` | Teacher-Student architecture |
| `teacher_mobilenetv2_backbone_b*.onnx` | `PaDiMFeatureExtractorAdapter` (calibration) + `PaDiMInferenceAdapter` (inference) | Shared backbones, NOT in model_path |

**`PaDiMInferenceAdapter` uses ONNX runtime** (not TFLite) for the backbone. It reads
`eval_config.json["backbone_paths"]` — a dict mapping `view_name → relative path` to the
ONNX backbone file (relative to `model_path`). Written automatically by
`CalibrationService.run_calibration()`. Example:
```json
"backbone_paths": {"front_view_section_1_A": "../backbones/teacher_mobilenetv2_backbone_b9.onnx"}
```
If `backbone_paths` is absent from `eval_config.json` (e.g. models calibrated before 2026-06-02),
update manually using the block number from `eval_config["blocks"]`.

**`PaDiMInferenceAdapter._normalize_mobilenet()`** applies ImageNet normalisation inside
`predict()` before running the ONNX backbone — exactly matching
`PaDiMFeatureExtractorAdapter._preprocess()` used during calibration. This is required because
the pipeline JSON does not include `normalize_mobilenet` for PaDiM sequences. Without this,
the backbone receives raw `[0, 255]` pixel values instead of the expected `[-2.1, 2.6]` range,
producing completely wrong features and causing every part to score as NOK.

**Critical rule**: backbone ONNX files go in `data/models/backbones/` — **never** in
`model_path`. The `AppFactory._build_inference_engine()` scans only `model_path`; it
will never encounter backbone files and will not misidentify the inference engine.

---

## Training pipeline — model export strategy

Training is done in PyTorch (see `Frambuesa/src/02-Pytorch/training_pytorch_float.py`).
The output is a `.pth` file. Before the inspection system can use the model,
it must be exported to TFLite. **Three export variants are needed** depending on
the target inference platform:

| File | Adapter that loads it | Target platform |
|---|---|---|
| `teacher_{view_name}_float32.tflite` | Both `CoralUsbInferenceAdapter` and `CpuInferenceAdapter` | All platforms |
| `student_{view_name}_int8_edgetpu.tflite` | `CoralUsbInferenceAdapter` | RPi + Coral USB |
| `student_{view_name}_float32.tflite` | `CpuInferenceAdapter` | PC / Jetson (no Coral) |

**Rule**: The teacher is always float32 for all platforms. The student has two variants:
- `_int8_edgetpu` for Coral (quantized, compiled with `edgetpu_compiler`).
- `_float32` for CPU/Jetson (no quantization needed — `tflite_runtime` runs it on CPU or GPU).

### Export script (pending — next training task)

A post-training export script must be written (`export_models.py`) that:
1. Loads the best `.pth` checkpoint.
2. Exports the **teacher** to `teacher_{view_name}_float32.tflite` via ONNX or `ai_edge_torch`.
3. Exports the **student** to `student_{view_name}_float32.tflite` (no quantization).
4. Exports the **student** to `student_{view_name}_int8_edgetpu.tflite` (int8 post-training
   quantization + EdgeTPU compilation via `edgetpu_compiler`).
5. Writes `eval_config.json` with `top_k_pixels`, `border_crop_px`, `gaussian_sigma`.
6. Writes `thresholds.json` with per-view `min`/`max` derived from the test set score
   distributions (formula used in training: `max = max(mean_OK + 5*std_OK, percentile_99.5_OK)`,
   `min = min(OK) * 0.90`).

All six output files go to the same `model_path` directory declared in the sequence JSON.

### AppFactory inference engine selection

`AppFactory._build_inference_engine()` auto-detects the engine by scanning the `model_path` directory (file-presence takes priority over `inference_device`):

| Files present in `model_path` | Engine selected |
|---|---|
| `padim_*_params.npz` | `PaDiMInferenceAdapter` |
| `student_*_int8_edgetpu.tflite` **or** `inference_device` contains `"Coral"` | `CoralUsbInferenceAdapter` |
| anything else | `CpuInferenceAdapter` (Teacher-Student float32) |

Deploying new model files to `model_path` is sufficient to switch inference engines — no sequence JSON edit needed.

### New sequence JSON fields (multi-platform)

Two new optional fields in `hardware`:

```json
"device_type": "RaspberryPi",   // "RaspberryPi" | "PC" | "Jetson" — selects GPIO adapter
"camera_index": 0               // OpenCV device index for USB cameras (default 0)
```

`device_type` defaults to `"RaspberryPi"` when absent (backward-compatible).
`camera_index` is only used when `camera_type != "CSI"`.

`AppFactory._build_hardware()` detects RPi with `"raspberry" in device_type.lower()` so
both `"RaspberryPi"` and legacy values like `"Raspberry Pi 4B"` are handled correctly.

---

## USB camera: stale-frame buffer behaviour (UsbCameraAdapter)

OpenCV `VideoCapture` maintains an internal FIFO queue that the kernel/V4L2 driver
fills continuously, regardless of whether anyone is reading. Calling `read()` returns
the **oldest frame in the queue**, not the current one. When previews are captured
on-demand (every few seconds), the queue can accumulate 5–10 stale frames, causing
the preview to look several seconds behind reality.

`UsbCameraAdapter` mitigates this with two layers:

1. `initialize_camera()` sets `CAP_PROP_BUFFERSIZE = 1` — requests the minimum internal
   buffer size from the driver. Not all V4L2 backends honour this.
2. `capture_frame()` calls `_cap.grab()` × `_BUFFER_DRAIN_FRAMES` (default 5) before
   calling `_cap.read()`. `grab()` advances the queue without decoding (cheap), ensuring
   the decoded frame is the most recent one.

`get_preview_frame_to_HTML()` does **not** drain the buffer because it is called at
stream rate (5 fps) and is always reading near-current frames anyway.

`CsiCameraAdapter` does not have this problem: Picamera2's `capture_request()` blocks
until the hardware delivers the next frame — it never returns a queued stale frame.

## USB camera: V4L2 resolution rounding (UsbCameraAdapter)

`cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT, w, h)` is a **hint** to the V4L2 driver — not a
guarantee. The driver silently rounds the requested value to the nearest sensor mode it
supports and returns success regardless. If the requested resolution is not in the
camera's firmware list, frames arrive at the native sensor resolution (e.g. 2304×1536
for a Logitech HD 1080p) even though `sequence.json` says 1280×720.

**This causes ROI coordinates to be wrong** — the canvas JS uses `captureW/captureH`
from the sequence JSON, but the actual frames have different dimensions.

**Fix implemented in `UsbCameraAdapter.initialize_camera()`**:
After `cap.set()`, the actual resolution is read back with `cap.get()`. If it differs
from the requested value, `_capture_resolution` is updated in place and a `[WARN]` is
printed. All callers (`capture_frame`, canvas scale factors) then use the real size.

**`AppFactory.get_capture_resolution() -> tuple[int, int]`**:
Returns the real capture dimensions after `initialize_hardware()`. Uses
`hasattr(_camera, "_capture_resolution")` (duck-typing — no interface change needed).

**`IrisServer._sync_capture_resolution(state)`** is called after every
`initialize_hardware()` in IrisServer (setup/finalize, save_sequence, _load_sequence):
- Updates `state.factory._sequence["hardware"]["camera_capture_resolution"]` in memory
  so `_build_view_configs()` and the calibration pipeline use correct coordinates.
- Updates `state.draft["hardware"]["camera_capture_resolution"]` in memory so the
  builder canvas JS receives the corrected resolution via the DRAFT template variable.
- Does **not** write to disk — the stored value reflects what the user configured.

**This is a USB-specific problem.** CSI cameras (Picamera2/libcamera) validate the
requested resolution and fail with an explicit error if not supported. USB cameras
always silently accept any resolution request.

---

## Abigail — Fleet Management Server (design phase)

Abigail is a **centralized web server** that monitors and controls **multiple
Raspberry Pi inspection systems** simultaneously. It runs off-device (typically
on a factory server or cloud instance) and provides plant managers and quality
engineers with a unified dashboard for fleet-wide operations.

### Architecture principles

1. **Iris retains authority** — each Raspberry Pi is the source of truth for its
   own state. Abigail does not store operational state; it queries and commands.
2. **REST over HTTP** — Abigail consumes the same Iris REST API that the local
   operator uses. No custom protocol needed.
3. **Polling-based sync** — Abigail polls `/api/status` on each registered Pi
   every 3 seconds. Future: upgrade to Server-Sent Events or WebSocket for push.
4. **Proxy pattern** — when a user clicks "Stop" in Abigail, it proxies the
   command as `POST http://{pi_ip}:5000/api/stop`. The Pi executes; Abigail
   observes the result on the next poll.
5. **Read-only heatmaps and logs** — Abigail can fetch `/api/heatmap/<view>` and
   parse traceability JSONL files over HTTP or via shared NFS mount.

### State synchronized (per Pi)

| Field | Source endpoint | Update frequency |
|---|---|---|
| `running` | `GET /api/status` | 3 s |
| `mode` | `GET /api/status` | 3 s |
| `sequence` | `GET /api/status` | 3 s |
| `label` | `GET /api/status` | 3 s (samples mode only) |
| `cycle_count` | `GET /api/status` | 3 s |
| `last_result` | `GET /api/last_result` | on-demand or when `cycle_count` increments |
| `available_sequences` | `GET /api/sequences` | on page load or manual refresh |

### Iris endpoints consumed by Abigail

Abigail acts as a remote client for the Iris API. All endpoints are already
implemented and tested locally; no changes to Iris logic required.

| Method | Path | Abigail use case |
|---|---|---|
| GET | `/api/info` | Pi metadata for discovery and monitoring (hostname, version, uptime) |
| GET | `/api/status` | Poll loop — 3 s interval per Pi |
| POST | `/api/start` | Fleet-wide start (parallel requests) |
| POST | `/api/stop` | Emergency stop (broadcast to all Pis) |
| POST | `/api/set_mode` | Switch Pi to samples mode remotely |
| POST | `/api/set_label` | Change label for training capture |
| POST | `/api/load_sequence` | Deploy a new sequence to a Pi |
| GET | `/api/sequences` | List available sequences on a Pi |
| GET | `/api/last_result` | Fetch inspection result for dashboard table |
| GET | `/api/heatmap/<view>` | Display anomaly heatmap in Abigail UI |
| GET | `/stream` | Embed live MJPEG stream in Abigail (iframe or img) |

### Pi registration in Abigail

Pis are registered in Abigail via a config file or database table. Each entry has:

```json
{
  "pi_id": "pi_nissan_line_a",
  "display_name": "Nissan Line A — Station 3",
  "ip_address": "192.168.1.101",
  "port": 5000,
  "location": "Plant 1 / Assembly Line A",
  "last_seen": "2026-05-27T14:32:15Z",
  "status": "online"
}
```

**Discovery options** (future):
- **Manual entry**: operator adds IP + label via Abigail UI
- **mDNS/Bonjour**: Pis broadcast `_iris._tcp.local` for auto-discovery
- **DHCP reservation**: static IPs assigned per MAC address

### Abigail page structure (proposed)

```
/                          ← Fleet dashboard (grid of Pi cards)
├─> /pi/<pi_id>            ← Single-Pi detailed view (mirrors Iris /inspection)
├─> /fleet/start           ← Bulk actions page
├─> /fleet/sequences       ← Sequence deployment wizard
├─> /reports               ← Aggregated statistics (OK rate, cycle time, etc.)
└─> /settings              ← Pi registration, polling interval, alerts
```

**Fleet dashboard card (per Pi):**
- Header: `pi_id` + status dot (green/red/yellow)
- Current sequence: `sequence_004.json (nissan_shroud)`
- Mode badge: `Inference` or `Samples (OK)`
- Running indicator: ● RUNNING / ○ STOPPED
- Cycle count: `#1,245 today`
- Last result: ✓ OK / ✗ NOK (score: 0.0021)
- Actions: [Start] [Stop] [View Details]

### Modifications required in Iris (implemented 2026-05-27)

#### 1. CORS headers (cross-origin requests) — ✅ IMPLEMENTED
When Abigail runs on a different domain/IP, browsers block AJAX requests due to
Same-Origin Policy. CORS middleware is now enabled in Iris:

```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})  # or restrict to Abigail IP
```

**Security note**: In production, restrict `origins` to the Abigail server IP or
use token-based authentication (JWT in `Authorization` header).

#### 2. `/api/info` endpoint — ✅ IMPLEMENTED
Returns static metadata about the Pi for Abigail's registration discovery:

```json
GET /api/info
{
  "hostname": "raspberrypi-line-a",
  "iris_version": "1.1.0",
  "device_type": "RaspberryPi",
  "uptime_seconds": 86400,
  "available_sequences": 3
}
```
```

#### 3. Optional: SSE endpoint for real-time push
Replace polling with Server-Sent Events for sub-second status updates:

```python
@app.route("/api/events")
def api_events():
    def event_stream():
        while True:
            with state.lock:
                data = {"running": state.is_running, "mode": state.current_mode, ...}
            yield f"data: {json.dumps(data)}\\n\\n"
            time.sleep(1)
    return Response(event_stream(), mimetype="text/event-stream")
```

Abigail client: `const eventSource = new EventSource("http://pi_ip:5000/api/events");`

#### 4. Current `/api/status` endpoint (already compatible)

The existing `/api/status` endpoint already returns all state needed by Abigail:

```json
GET /api/status
{
  "running": true,
  "mode": "inference",
  "dry_run": false,
  "sequence": "sequence_004.json",
  "label": null,
  "cycle_count": 1245
}
```

No changes needed — Abigail can consume this directly. The `label` field is
populated only when `mode == "samples"` (e.g., `"ok"` or `"nok"`).

### Synchronization guarantees

| Event | Propagation |
|---|---|
| Operator presses Stop on Iris locally | Abigail sees `running=false` within 3 s (next poll) |
| Manager presses Stop on Abigail | Pi stops immediately; Abigail confirms on next poll |
| Sequence changed via Iris `/inspection` page | Abigail sees new `sequence` name within 3 s |
| Mode switched via Abigail | Pi switches mode; Abigail confirms within 3 s |
| Inspection cycle completes | `cycle_count` increments; Abigail fetches `last_result` |

**Conflict resolution**: If the Pi and Abigail both issue commands simultaneously
(e.g., both try to stop), the Pi's local command wins (it arrives first). Abigail's
command may return HTTP 409 if the state changed between poll and command.

### Abigail tech stack (proposed)

| Layer | Technology |
|---|---|
| Backend | Flask or FastAPI (Python 3.10+) |
| Frontend | Jinja2 templates + Alpine.js or HTMX for reactive UI |
| State | In-memory dict per Pi (no database for MVP) |
| Polling | `threading.Timer` or `asyncio.create_task` per Pi |
| Auth | Flask-Login + session cookies (optional JWT for API) |
| Deployment | gunicorn + systemd service (same as Iris) |

**File structure** (mirrors Iris):
```
Abigail/
├── abigail/
│   ├── __init__.py
│   ├── AbigailServer.py       ← Flask app, fleet endpoints
│   ├── PiRegistry.py          ← Pi registration + polling orchestrator
│   ├── templates/
│   │   ├── dashboard.html     ← Fleet grid view
│   │   └── pi_detail.html     ← Single-Pi drilldown
│   └── static/
│       ├── css/abigail.css
│       └── js/dashboard.js    ← Status polling + card updates
├── config/
│   └── pi_registry.json       ← Registered Pis (or SQLite DB)
└── main.py
```

### Testing synchronization locally

Simulate a 2-Pi fleet on one machine:

```bash
# Terminal 1: Pi A on port 5000
cd InspectionApp
python3 -m iris.IrisServer

# Terminal 2: Pi B on port 5001
cd InspectionApp_clone
flask --app iris.IrisServer:create_iris_app run --port 5001

# Terminal 3: Abigail
cd Abigail
python3 -m abigail.AbigailServer

# Register both in config/pi_registry.json:
[
  {"pi_id": "pi_a", "ip_address": "127.0.0.1", "port": 5000},
  {"pi_id": "pi_b", "ip_address": "127.0.0.1", "port": 5001}
]
```

Open `http://localhost:8000` → should see 2 Pi cards with independent state.

---

## Pending work (not yet implemented)

| Item | Notes |
|---|---|
| **`export_models.py`** | ⏸ **PAUSED** — Teacher-Student architecture is not in use for the foreseeable future. PaDiM is the active inference engine. Revisit only if Teacher-Student is reactivated. |
| **`DatabaseAdapter`** | Future `IRepository` implementation that writes to a relational DB. The JSONL schema is already normalized to 4 tables. |
| **Iris: sequence clone** | ~~Implemented 2026-07-31~~ — new "Clone as new part →" button on the Home page (next to "Edit sequence"), opens a modal (`.modal-overlay`/`.modal-box` pattern, same as the Restart Iris modal) asking for a new `part_model` name, with the same real-time duplicate check as the setup wizard (`POST /api/validate_part_model`, debounced 500 ms). Confirming calls the new `POST /api/clone_sequence {path, new_part_model}`: copies the source sequence's hardware/pipeline/steps **verbatim** into the draft (the whole point — reuse a similar part's setup instead of rebuilding from scratch in the builder), but always regenerates `part_model` and `paths` for the new part via a shared helper `_generate_paths_for_part_model()` (extracted from `/api/setup/finalize`, both routes now call it) — a clone is a physically different part and must never share images/model directories with its source. The draft is tagged with `_cloned_from` (source filename, UI-badge only) instead of `_edit_path`, so the builder's existing "Save Sequence" button treats it like any brand-new sequence and writes a new auto-numbered `sequence_NNN.json` — **zero changes needed to the save path itself**, `_cloned_from` is stripped before persisting (mirrors how `_edit_path` is already stripped in edit mode). `builder.html` shows a "Cloned from: ..." badge (alongside the existing "Editing: ..." badge) so the operator knows they're building a new sequence, not overwriting the source. Validated with `py_compile` only (no local venv with pytest in this environment). |
| **Bugfix: Home page unreachable once a sequence exists** | ~~Fixed 2026-07-31~~ — found right after shipping the clone feature above: `home()` redirects `/` → `/inspection` whenever `list_sequences()` is non-empty (existing behavior since before this session, meant to skip the picker for daily operators), which in practice is **always true** in production after initial setup — so the picker page holding "Edit sequence"/"Clone as new part" had become permanently unreachable through normal navigation, including via the top-nav "Home" link (it just bounced straight back to Inspection). Fix: `home()` now accepts `?manage=1` to bypass the redirect and always render the picker; `base.html`'s "Home" nav link was updated to `url_for('home', manage=1)` so operators can actually reach it by clicking Home. Plain visits to `/` (e.g. the kiosk browser's start URL in `start_iris.sh`) are unaffected and still redirect straight to Inspection. |
| **RPi auto-start + watchdog** | ~~Implemented 2026-07-31~~ — auto-start was already in place (`setup/start_iris.sh` + `setup/iris-browser.desktop`, 2026-06-02). The external watchdog (`setup/iris_watchdog.sh` + `setup/iris_watchdog.service`) polls `/api/info` every `IRIS_WATCHDOG_POLL_INTERVAL_S` (default 10s, env-configurable), and after `IRIS_WATCHDOG_FAILURE_THRESHOLD` (default 3) consecutive misses, recovers automatically. **Distinct from the in-process cycle watchdog** (`_cycle_watchdog()` in `IrisServer.py`, row above) — this one runs entirely OUTSIDE the Python process, so it also catches a deadlocked interpreter or a hung gunicorn worker that never even reaches the in-process watchdog's own checks. **No root/sudo required anywhere** — recovery only ever kills/restarts processes already owned by the same `pi` user, deliberately avoiding the sudoers `NOPASSWD` workaround style rejected earlier for the full-OS-reboot feature (see "Reboot button on Iris home page" row). Detects which of the project's two deployment models is active on every check and reacts accordingly: **Model A** — `iris.service` active (`systemctl is-active --quiet iris`) → just kills the hung `gunicorn` process; systemd's own `Restart=on-failure` notices the exit and restarts it, exactly like the in-process watchdog's `os._exit(1)` already relies on. **Model B** — no `iris.service` unit (kiosk Pi using `iris-browser.desktop` → `start_iris.sh` directly, no systemd unit for Iris itself) → kills the hung `gunicorn` process AND the stale kiosk browser window, then relaunches `start_iris.sh` in the background (which starts gunicorn again and reopens the browser). `iris_watchdog.service` runs as `User=pi` (same user as `iris.service`) with `DISPLAY=:0`/`XAUTHORITY` set so it can manage the kiosk browser's X11 window in Model B (harmless no-ops in Model A, which never touches a browser). Installation documented as new "11. External watchdog" section in `commands_first_instalation_rpi.txt`. Validated with `bash -n` only — full validation requires a real Pi with both deployment models. |
| **Emergency stop button** | ~~Implemented 2026-06-02~~ — red `⏹ STOP` in `base.html` topbar, calls `POST /api/stop`. |
| **Iris: sample review & relabeling page** | ~~Implemented 2026-07-31~~. Replaced **Step 1 — Capture Images** on the calibration page with a **review/relabel gallery** (same `section-capture` id in `calibration.html`, retitled "Step 1 — Review Images"; the old `train_ok`/`test_ok`/`test_nok` radio group, Start/Stop Capture buttons, and Live Stream/Last Captures preview tabs were removed from the template). 4 tabs (`data-set` attribute, reusing the existing `.cal-tab`/`.cal-tab-active` CSS): `train_ok`, `test_ok`, `test_nok`, `discarded` — plus a per-view `<select>` (reusing the `view_sections` Jinja variable already passed by the `calibration()` route) and a paginated `.gallery-grid` (reusing the existing gallery CSS from the production-review modal). New pure core service `CaptureReviewService` (`app/src/core/services/CaptureReviewService.py`, no hardware dependency): `list_images(target_set, view_name, page, page_size)` and `relabel_images(paths, target_set)` — the latter validates every path resolves inside `images_path` (`os.path.realpath` check) before moving anything, and moves files via `os.replace()`, so discarding is always reversible (no destructive delete). New `AppFactory.create_capture_review_service()` (mirrors `create_traceability_review_service()`). Two new routes in `IrisServer.py`: `GET /api/review/images?set=train_ok|test_ok|test_nok|discarded&view=...&page=N&page_size=N` and `POST /api/review/relabel {paths: [...], target_set: "..."}`. `api_calibration_image_counts()` extended with a `"discarded"` entry so the existing `os.walk()`-based counting picks it up automatically — `#count-discarded` badge added next to the existing 3 count badges. `calibration.js` fully rewritten for this section: all old capture/stream/tab-switching functions removed, replaced by `loadReviewGallery()` / `renderReviewGallery()` / `relabelImage()`, reusing the `/api/image_file?path=` endpoint (already validates paths against `data/`) for thumbnails. **Decision: the OLD calibration-capture backend is left as unremoved, flagged dead code** (not deleted this session, to limit risk without real-Pi hardware validation) — `cal_capture_controller`, `/api/calibration/set_target|start_capture|stop_capture`, the capture-fallback branch in `/stream`, and the `_camera_keepalive` thread's capture-active check in `IrisServer.py` are no longer reachable from any UI (confirmed the old `start_capture` route only ever instantiated a temporary `GuiSamplesAdapter`+`SampleCaptureService`, identical to what Samples mode already does) — safe to remove in a future low-risk cleanup pass. |
| **Iris: Samples mode gains a 3-way label (replaces calibration's old capture step)** | ~~Implemented 2026-07-31~~. Added `"test_ok": os.path.join("test", "OK")` to `SampleCaptureService._LABEL_SUBDIRS` (keeping `"ok"` = train/OK unchanged) — `count_samples()`/`count_all_labels()` needed no changes since they already iterate `_LABEL_SUBDIRS` generically. `inspection.html`'s `#label-group` now renders **two** `.toggle-group` divs instead of one: a solo **Train OK** button, then a **Test OK**/**Test NOK** pair — the visual split (rather than one 3-button group) satisfies the confirmed requirement of extra spacing between Train OK and the Test OK/Test NOK pair. New `.label-btn-trainok` CSS class in `iris.css` (soft/light green tint, distinct from the primary-blue `.toggle-btn.active` and from the NOK/error red family) gives Train OK its required visual differentiation. No `inspection.js` changes were needed — the existing generic `.label-btn`/`data-label`-based binding already works unchanged with the new third value. |
| **Iris: NOK-only inference image persistence** | ~~Implemented 2026-07-31~~. **Design finalized 2026-07-31 — fixed behavior, no per-sequence toggle** (user confirmed simplicity over configurability), motivated by the review/discard page above and by SD-card read/write wear. Currently `InspectionService.save_results()` → `LocalStorageAdapter.save_frames()` writes **every captured view of every cycle** (OK and NOK alike) to `{inference_images_path}/YYYYMM/YYYYMMDD/...`. Change: only persist frames for views whose `InspectionResult.is_ok is False` (per-view filtering, not per-part) — build the set of NOK `view_name`s from `part.inspection_results` and filter `captured_frames` before calling `_repository.save_frames()`. Rationale: (1) keeps the system's scope clearly bounded — the disk-based review/promote pipeline (`TraceabilityReviewService`, the review/discard page, "Recalibrate from Production") already only ever operates on NOK images, so OK inference frames were never used for anything; (2) cuts continuous SD-card writes roughly in proportion to the OK rate, which is the dominant fraction in a healthy line. `images_path/latest/{view_name}.jpg` (`GuiInferenceAdapter._save_latest_frames()`, live preview fallback) is a **separate mechanism and must keep saving every cycle** — it is a single overwritten file per view, not a growing dataset, and the live preview must work regardless of OK/NOK. Do not apply the NOK-only filter there. |

**Confirmed implementation order (2026-07-31, user-approved)**: (1) camera reconnection robustness — ~~implemented 2026-07-31~~, see "Camera reconnection robustness" row below, (2) sequence indicator on the calibration page — ~~implemented 2026-07-31~~, see row below, (3) the review/discard viewer + Samples 3-way label (incl. Train OK visual differentiation) + NOK-only persistence — ~~implemented 2026-07-31~~, see the three rows above, (4) Schedule Timed Captures — ~~implemented 2026-07-31~~, see row below, (5) Reboot button on Iris home page — queued, see "New feature requests" below.
| **Iris: sequence indicator on calibration page** | ~~Implemented 2026-07-31~~ — `calibration.html` now shows the same "Sequence" `<select>` + "Load" button as `inspection.html` (`#cal-seq-select`/`#btn-cal-load-seq`, inside `.cal-header`, always rendered even with no sequence loaded). `calibration()` route in `IrisServer.py` passes `sequences`/`current_sequence` the same way `inspection()` does. `calibration.js`'s `loadSequence()` reuses the existing `POST /api/load_sequence` endpoint (no backend change), applies the same loading-indicator pattern as `inspection.js` (disables select+button, shows "Loading…"), and on success calls `window.location.reload()` — simpler and safer than patching `best_blocks`/image counts/view sections individually via JS, since all of those are computed server-side in the `calibration()` route. |
| **Camera reconnection robustness — align crash recovery with Load Sequence** | ~~Implemented 2026-07-31~~ — `POST /api/load_sequence` does a full hardware reset (new camera **and** MUX), but the old per-cycle crash-recovery handler in `GuiInferenceAdapter._run_loop()` only reset the camera, never the MUX — a stale MUX/GPIO channel state was never cleared. Fix (kept architecture-compliant, see invariant #4): added `ICsiMux.reinitialize()` (`RpiCsiMuxAdapter` redoes the GPIO double-init→close→re-init cycle and resets `_current_channel = None`; no-op in `NoOpMuxAdapter`) and `SequenceExecutor.recover_hardware()` (`camera.close_camera()` → `camera.initialize_camera()` → `mux.reinitialize()`), called from `_run_loop()`'s crash-recovery block instead of touching `self._camera` directly. Validated with `py_compile` only — full validation still requires the real Pi. **Rollback flag added 2026-07-31**: module-level `_USE_FULL_HARDWARE_RECOVERY = True` in `GuiInferenceAdapter.py` — set to `False` to revert to the old camera-only recovery (both code paths coexist) until the new behavior is validated on real multi-camera hardware; remove the flag once confirmed. **User-facing feedback added 2026-07-31**: `GuiInferenceAdapter._stopped_reason` (surfaced via `get_status()` → `/api/status` → `inspection.js`) shows a persistent red banner ("Camera/hardware initialization error…") when the loop stops itself after `_MAX_CONSECUTIVE_ERRORS` (5) consecutive failures — cleared automatically on the next `start_loop()`. **Loading indicator added 2026-07-31**: `inspection.js`'s `#btn-load-seq` click handler now disables the button + `#seq-select` and shows "Loading…" while `POST /api/load_sequence` is in flight, then a green success toast or red error toast — prevents multi-click freezes during the multi-second hardware reinit. The 5-retry cap was already correct in two places (`CsiCameraAdapter.MAX_RETRIES = 5`, low-level camera init; `GuiInferenceAdapter._MAX_CONSECUTIVE_ERRORS = 5`, loop-level) — only the user-facing surfacing was missing. Same loading-indicator pattern should be reused for the calibration page's sequence Load button once that row (below) is implemented. |
| **In-process cycle watchdog** | ~~Implemented 2026-07-31~~ — a single daemon thread (`_cycle_watchdog()` in `IrisServer.py`, started in `create_iris_app()` alongside the existing camera-keepalive thread) polls every 5 s and force-exits (`os._exit(1)`) if the active inspection cycle has been running longer than `cycle_watchdog_timeout_s` (`config/default_values.json`, default 60 — read fresh on every check). systemd's existing `Restart=on-failure, RestartSec=5s` (`setup/iris.service`) then brings the process back up, and the session-state auto-resume feature (see row above) restarts the inspection loop automatically. **Crucially, the timer only starts once the physical trigger is actually received, not when the cycle's `wait_for_input` step begins waiting for it** — that wait is unbounded by design and can legitimately last hours between production runs. `SequenceExecutor` tracks this with `_live_cycle_started_at`/`_cycle_timer_lock`, set at the same moment as `part._actual_start_time` (inside the `wait_for_input` branch of `_execute_gpio_action`, only when `timeout_ms` is `0`/`None` and the signal was received) and cleared to `None` at the end of `run()`. `get_live_cycle_age_s()` (public on both `SequenceExecutor` and, via passthrough, `GuiInferenceAdapter`) returns `None` while idle/paused/waiting-for-trigger, or the elapsed seconds since the real trigger otherwise — the watchdog thread only acts when this is a number greater than the timeout. **Bugfix 2026-08-05 (false-positive restart while still waiting for a trigger)**: sequences that chain more than one indefinite `wait_for_input` step (e.g. `sequence_001.json`'s "Esperar inicio de ciclo" step 4 immediately followed by "Esperar pieza 1ra pos" step 5, both `timeout: 0`) used to only *set* `_live_cycle_started_at` on each successful indefinite wait, never *clear* it before blocking on the next one — so the timer kept ticking from step 4's success straight through step 5's (legitimately long) wait for the next physical piece, and the watchdog force-restarted the process mid-wait once that exceeded `cycle_watchdog_timeout_s` (observed in production on the `192.168.45.201` Pi: killed after 60.6 s stuck in "Esperar pieza 1ra pos"). Fixed in `_execute_gpio_action()`: right before blocking on any `wait_for_input` with `timeout_ms` `0`/`None`, `_live_cycle_started_at` is now explicitly set back to `None` (pausing the watchdog), and only set to `time.monotonic()` again once that specific wait succeeds — so the watchdog is paused for the full duration of every indefinite trigger wait in the sequence, not just before the first one. `cycle_watchdog_timeout_s` is intentionally **not exposed anywhere in the Iris web UI** (no settings route reads it) — editable only by hand-editing the JSON file, for the rare case a real cycle needs more than 60 s (observed real-world max so far: ~15 s trigger-to-inference-result, including robot movement time). `gunicorn.conf.py` `timeout=0` remains unchanged and is unrelated — it governs the HTTP worker, not this cycle-level check. |
| **Iris: persisted session-state auto-resume** | ~~Implemented 2026-07-31~~. Scoped to **inference mode only** — Samples mode capture sessions are never auto-resumed (choosing the resume label is an operator decision). `IrisServer.py`: new `config/session_state.json` (best-effort, written/removed by `_save_session_state()`/`_clear_session_state()`). `api_start()` calls `_save_session_state(seq_path)` when `state.current_mode == "inference"` right after `ctrl.start_loop()` succeeds (or `_clear_session_state()` if starting in Samples mode). `api_stop()` always calls `_clear_session_state()` — an operator-initiated stop must never be auto-resumed. `_resume_session_if_any(state)` runs once at the very end of `create_iris_app()` (right before `return app`): if `config/session_state.json` exists and its `sequence_path` still points to a real file, it calls `_load_sequence()` + `ctrl.start_loop()` synchronously during app boot. This means a crash, a systemd `Restart=on-failure` cycle, or the new "Restart Iris" button all bring the inspection loop back up automatically — an unattended production line does not stay halted just because the web service had to restart. All I/O is wrapped in `try/except` (best-effort; a corrupt/missing state file or a hardware init failure at boot must never crash the whole app). |
| **`commands_first_instalation_rpi.txt`: systemd + kiosk autostart sections** | ~~Implemented 2026-07-31~~. The file previously ended at camera config + `sudo reboot` with **no** section at all for installing `iris.service` — added two new numbered sections after it: "9. Install Iris as a systemd service" (`sudo cp setup/iris.service /etc/systemd/system/`, `daemon-reload`, `enable`, `start`, plus `status`/`restart`/`journalctl -u iris -f` as reference commands — mirrors the commands already documented in `setup/iris.service`'s own header and in this file's "Running Iris" section above) and "10. Kiosk auto-launch" (`cp setup/iris-browser.desktop ~/.config/autostart/`, noted as optional/only for Pis with an attached monitor — mirrors `setup/start_iris.sh`'s own header comment). |

### New feature requests (design confirmed 2026-07-31, not yet implemented)

| Item | Notes |
|---|---|
| **Schedule Timed Captures** | ~~Implemented 2026-07-31~~ — automates periodic dataset collection while Samples mode keeps running on real production triggers, without switching label manually for every cycle. New "🕒 Schedule" button in the Inspection page's Label group (`inspection.html`), visible only in Samples mode and only when the Train OK or Test OK label is selected (Test NOK excluded — NOK occurrences are expected to be manually reviewed, not batch-scheduled); opens a modal (`#schedule-modal`) with images-per-window, interval (hours + minutes), and target-images fields (inline warning above 500). Confirmed design answers: (1) unit is production cycles, not individual files — matches how training sets are built per part; (2) destination label is the Train OK / Test OK label the operator had selected when enabling the schedule (snapshotted, not re-read live); (3) state is in-memory only, never persisted to disk (lost on restart, same as `dry_run`). All scheduling state/logic lives inside `SampleCaptureService` (not a second service) to respect the "input adapters hold only one service" invariant: `enable_schedule()`, `disable_schedule()`, `is_schedule_enabled()`, `get_schedule_status()`, `next_cycle_plan(manual_label)` (decides per-cycle whether to persist and with which label, opening/closing capture windows every `interval_s`, auto-pausing persistence once `target_images` total cycles are persisted while staying armed), and `schedule_record_persisted()`. `run_capture_cycle()` gained a `persist: bool = True` param; the always-on `latest/{view_name}.jpg` live-preview write (used by the Inspection page's frame poller) was moved from `GuiSamplesAdapter._save_latest_frames()` (now removed, along with its now-unused `images_path` constructor param) into a new `SampleCaptureService._save_latest_preview()`, called unconditionally every cycle (regardless of `persist`) directly on the in-memory frame instead of round-tripping through disk. `GuiSamplesAdapter._run_loop()` calls `next_cycle_plan()` then `run_capture_cycle(label, persist=persist)` then `schedule_record_persisted()` when persisted. New Flask routes `POST /api/schedule/enable` (validates positive numeric inputs, samples mode, and label \in {ok, test_ok} as defense-in-depth against the UI-only gating) and `POST /api/schedule/disable`; `/api/status` now includes a `schedule` object (`null` outside samples mode). |
| **Reboot button on Iris home page** | ~~Implemented 2026-07-31 (as "Restart Iris", process-level restart — not a full OS reboot)~~. After reconsidering the OS-reboot design below, a simpler and safer alternative was implemented instead: a "⟳ Restart Iris" button in the **topbar** (`base.html`, next to `⏹ STOP` — NOT on the Home page: `/` redirects straight to `/inspection` whenever at least one sequence exists, per the `home()` route in `IrisServer.py`, so a Home-page-only button would be unreachable in any real deployment) opens a confirmation modal (reusing the `.modal-overlay`/`.modal-box` pattern, rendered once in `base.html` so every page has it) asking for a PIN, then calls `POST /api/system/restart_iris`. The route checks the PIN against `restart_pin` in `config/default_values.json` (read fresh on every call, so editing the file takes effect immediately, no code change or restart-before-the-restart needed; default value `"2679"` — **not a real credential** — it only prevents accidental clicks, since this endpoint needs no elevated privileges at all), rejects with 409 if `state.is_busy` (the loop must be stopped first, so `os._exit(1)` never kills a cycle mid-flight and leaves a GPIO output stuck high), then spawns a daemon thread that sleeps 0.5 s (to let the HTTP response flush) and calls `os._exit(1)` — systemd's existing `Restart=on-failure, RestartSec=5s` (`setup/iris.service`) brings the process back up automatically within a few seconds, no sudoers/root privileges needed. `base.html`'s JS polls `/api/info` every 1.5 s after triggering the restart and reloads the page automatically once it responds again. **The full OS-level `sudo reboot` variant (with the sudoers `NOPASSWD` workaround) described below was NOT implemented** — it remains a separate, more complex feature to revisit only if a full Pi reboot (not just restarting the Iris process) is ever actually needed; original design notes kept for reference: |

### Future vision features (not yet scoped)

- **Color-based detection**: define rectangular or custom-polygon areas in the Iris
  builder and run color-distribution checks (histogram matching, mean hue threshold, etc.)
  entirely by vision — no neural model required. Each area would have its own
  color acceptance rule stored in the sequence JSON.
- **Custom polygon mask for PaDiM**: let the operator draw an arbitrary polygon
  (defined by a set of vertices) in the Iris builder canvas. The polygon would be
  serialized as a `put_black_polygon` pipeline tool and applied as a mask before
  inference, similar to `put_black_circle` / `put_black_rectangle`.
- **Drift detection and recalibration recommendation**: the system should monitor
  whether the separation between OK and NOK score distributions is shrinking over
  time. A separation ratio < 1.2x (i.e. `min(NOK) / max(OK) < 1.2`) is a signal
  that the model is drifting — likely because product/lighting conditions have
  changed since calibration.

  **Design sketch**:
  - After each inspection cycle, `LocalStorageAdapter` already saves every score
    to the JSONL traceability file. A background job (or periodic Iris endpoint)
    can compute the rolling `max(OK_score)` over the last N confirmed-OK parts.
  - When `min_nok_recent / max_ok_rolling < 1.2`, Iris shows a persistent banner:
    _"Recomendación: Recalibrar sistema — La separación OK/NOK ha caído por debajo
    de 1.2x. Capture imágenes OK recientes, verifique manualmente que sean correctas
    y ejecute el pipeline de entrenamiento."_
  - **Human-in-the-loop requirement**: the system cannot retrain automatically
    because the captured inference images may include genuine NOK parts that were
    incorrectly accepted. An operator must visually review the candidate images
    before they are moved to the training set.
  - Suggested thresholds: warn at < 1.2x, critical at < 1.05x (near-overlap).
  - The `eval_config.json` already stores the original separation from training;
    this value is the reference baseline for drift comparison.
  - Implementation touches: `LocalStorageAdapter` (read recent scores),
    `IrisServer.py` (new `/api/drift_status` endpoint + banner logic in
    `inspection.html`), and potentially a lightweight `DriftMonitorService` in
    `core/services/`.

- **Automatic recalibration dataset builder**: when drift is detected and the
  operator confirms recalibration is needed, the system should assist in building
  the new training set from existing inference images. Rules:
  - **Source**: `images_path/inference/{channel}/` (images already inspected and
    accepted as OK by the current model). Must be human-reviewed before use.
  - **Selection algorithm** (to avoid Gaussian bias from date imbalance):
    1. Group images by capture date (extracted from filename timestamp).
    2. If any single date has > 20 images and dominates the set, keep only 20
       images from that date, preferring images spread across different hours.
    3. If total count > 500, remove images from the dates with the most images
       until the distribution across dates is roughly uniform.
    4. Target: 200–500 images, no single date representing > 30 % of the total.
  - **Action**: copy selected images to `images_path/train/OK/{channel}/`
    (do NOT move — inference copies are retained as audit trail).
  - **Then**: trigger the normal calibration sweep pipeline from Iris.
  - **Algorithm location**: `core/services/` as a new `DatasetBalancerService`
    or as a static helper in `CalibrationService`.
  - **Iris touchpoints**: new `/api/calibration/suggest_retraining_dataset`
    endpoint (returns counts + date distribution), a "Prepare dataset" button
    in the calibration page, and a confirmation dialog before copying.

---

## Web server design (Iris — implemented 2026-05-20)

Iris is a Flask multi-page app running **on the Raspberry Pi** itself.
It is the local operator interface: the machine operator uses it to configure
a product sequence, start/stop the inspection loop, capture training samples,
and view live results.

**Abigail** is a separate, future system — a fleet-level server running
off-device that connects to multiple Pi units simultaneously. It shares the
same endpoint contract as Iris but is a different project.

### Page flow

```
Boot
 └─> /                       ← Home — select sequence or go to setup
      ├─> /inspection         ← Operational page (inference / samples / dry run)
      ├─> /calibration        ← Calibration page (capture → sweep → calibrate → deploy)
      └─> /setup              ← 3-step hardware wizard (first-time or new sequence)
           └─> /builder       ← Sequence builder: canvas tools + step editor
                └─> /inspection
```

Pages `/setup` and `/builder` are the **configuration path**, not the daily
operator path. The normal operator only sees `/` and `/inspection`.

### File location
```
InspectionApp/
└── iris/
    ├── __init__.py
    ├── IrisServer.py          ← Flask app, all routes, create_iris_app()
    ├── IrisState.py           ← Thread-safe state container
    ├── templates/
    │   ├── base.html          ← Shared layout, topbar, status dot
    │   ├── home.html          ← / — start inspection or new sequence
    │   ├── setup.html         ← /setup — 4-step hardware wizard
    │   ├── builder.html       ← /builder — canvas tools + step editor
    │   └── inspection.html   ← /inspection — operational interface
    └── static/
        ├── css/iris.css
        └── js/
            ├── fabric.min.js  ← Fabric.js (must be downloaded manually — no CDN)
            ├── canvas_tools.js
            ├── builder.js
            └── inspection.js
```

### New backend file
`app/src/core/services/SequenceBuilderService.py` — core service that builds
and validates sequence JSON from wizard state. Methods:
- `create_draft(hardware_config)` → initial draft dict
- `update_pipeline(draft, camera_port, view, section, pipeline)` → updated draft
- `add_step(draft, step)` / `remove_step(draft, step_number)` → updated draft
- `validate(draft)` → `list[str]` (empty = valid)
- `save_draft(draft, path)` / `load_draft(path)` → disk persistence
- `save_sequence(draft, sequences_dir)` → validated final JSON file
- `list_sequences(sequences_dir)` → `list[dict]` with path/filename/part_model/modified

### New AppFactory methods added for Iris
- `get_camera_ports() -> list[str]` — returns the camera port list from sequence
- `capture_preview_frame(channel) -> numpy.ndarray` — selects MUX channel, captures
  one full-resolution RGB frame, returns it as numpy array for JPEG conversion

### State held by Iris (IrisState)

```python
factory:                    AppFactory | None
controller:                 GuiInferenceAdapter | GuiSamplesAdapter | None
current_mode:               str                  # "inference" | "samples"
dry_run:                    bool                 # True → skip NOK GPIO steps (step_number < 0)
current_sequence_path:      str | None
captured_frames:            dict[str, bytes]     # channel -> JPEG bytes
draft:                      dict | None          # sequence draft in builder
calibration_controller:     GuiCalibrationAdapter | None
calibration_target:         str                  # "train_ok" | "test_ok" | "test_nok"
calibration_progress:       dict | None          # {step, total, message, done, error}
lock:                       threading.Lock
```

### IO module catalog
`config/io_module_catalog.json` documents the 3 available IO modules.
All 3 versions use the same RPi GPIO pins:
- Input BCM pins (connected to module IN ports): 13, 19, 26 (physical 33, 35, 37)
- Output BCM pins (connected to module OUT ports): 16, 20, 21 (physical 36, 38, 40)

v2 and v3 add a per-channel relay mode switch. In relay mode each output drives
a relay coil with contact pairs (e.g. OUT1→OUT1R1, OUT1R2) with NA/COM/NC terminals.

### Setup wizard (3 steps)
1. **Device**: part model name (validated in real-time against saved sequences), device
   type (from `device_catalog.json`), inference accelerator, IO module. The IO module
   selector is hidden when `supports_gpio=false` (e.g. PC). Inference and camera-type
   options filter automatically per device. Part model validation excludes
   `sequence_draft.json` from duplicate checks.
2. **Cameras**: type (CSI/USB — filtered by device), model (from `camera_catalog.json`),
   count (capped by `max_cameras`), quality preset (Alta/Media/Baja → `resolutions` values).
   Step 2 Next button calls finalize directly when `supports_gpio=false`.
3. **GPIO** (RPi/Jetson only — skipped for PC): Physical pin tables from the IO module.
   - **Input table**: one row per IN pin showing physical pin number, BCM, description,
     and an "Es trigger" checkbox. Multiple pins can be marked as triggers.
   - **Output table**: one row per OUT pin showing physical pin number, BCM, description,
     and a use dropdown (No usar / Señal PLC / Reflector).
   - Step 3 finalize button calls `POST /api/setup/finalize` which generates paths
     automatically based on `part_model`, creates draft, initializes hardware, and
     redirects to `/builder`.

**Path auto-generation**: All storage paths are generated automatically in
`/api/setup/finalize` using the pattern `./data/{category}/{part_model}/...`:
- `images_path`: `./data/images/{part_model}/`
  - Subdirectories created on first capture: `train/OK/`, `test/OK/`, `test/NOK/`
  - `inference/` is created by `LocalStorageAdapter` on first save
  - `latest/` is created by `GuiInferenceAdapter._save_latest_frames()` on first cycle
- `inference_images_path`: `./data/images/{part_model}/inference/`
- `traceability_inference_path`: `./data/traceability/{part_model}/inference/`
- `model_path`: `./data/models/{part_model}/`

Users cannot manually edit paths — this prevents configuration errors and ensures
consistent directory structure across all sequences.

### Builder page
Three-panel layout:
- **Left**: section tabs + step tree. Sections correspond to preprocessing
  pipeline groups. Steps are added/edited via a modal editor. Each step shows
  edit (✏) and delete (✕) icons.
- **Centre**: camera grid (1–4 canvas cards with Fabric.js overlay) +
  tool palette (ROI, black circle, black rectangle, **Detection ROI** (orange), select, delete).
  "Update Preview" button captures live frames from all cameras.
  The "Detection ROI" tool draws an orange rectangle that defines the `detect_piece_action` ROI.
  It is NOT a preprocessing pipeline tool — it is stored in the `detect_piece` step data.
  Only one Detection ROI per sequence is allowed (validated by `SequenceBuilderService`).
- **Right**: pipeline list for selected camera/section + camera settings
  (exposure, lens) + resize tool. Pipeline parameters display responsively
  with automatic wrapping.

Canvas coordinate scaling: all coordinates are stored in original capture
resolution. On export, canvas coords are multiplied by `captureW/displayW`
and `captureH/displayH`.

### Sequence constraints enforced by SequenceBuilderService.validate()
- `part_model` not empty
- `number_of_cameras` matches `camera_port` list length
- Every pipeline has `resize_to_training_resolution` with width and height
- At least one step with `step_number >= 1001` (inference)
- At least one step with `step_number < 0` (NOK dispatch)
- At most one step with `detect_piece_action` (only one piece detection per sequence)
- All GPIO pins in steps are declared in `gpio_configuration` (conditional: only enforced when `device_type != "PC"` or `gpio_configuration` is non-empty)
- All camera ports in steps are declared in `hardware.camera_port`

### Inspection page
- Sequence selector + Load button
- Mode toggle: Inference / Samples (rejected while loop running)
- **Dry Run toggle**: when enabled, `SequenceExecutor.run()` is called with `dry_run=True` —
  NOK GPIO steps (step_number < 0) are skipped. Indicator badge shown while active.
  Rejected while loop is running. State stored in `IrisState.dry_run`.
- Label toggle OK/NOK (samples mode only)
- Start / Stop buttons
- Preview button (on-demand, loop idle only)
- MJPEG stream (active only while loop is running; lazy — stops when loop stops)
- Last result panel: overall OK/NOK badge + `piece_detected` indicator, per-view score table, heatmap links, and (since section 23) a `failed_channel`-based callout ("⚠ Camera X failed: ...") shown next to the `ERROR_ABORTED` badge, plus the matching camera card gets the same `disconnected` badge used by the preview flow.
- **View mode tabs**: two tabs above the camera grid — **Live Stream** (`/stream` MJPEG, active while loop runs; switches to frame poller if "Last Captures" tab is active when loop starts) and **Last Captures** (`/api/frame/{ch}` polled every 2.5 s via `fetch→blob→createObjectURL` to avoid flicker; shows last inference frame per channel from RAM, or disk fallback from `images_path/latest/` after restart, or last manual preview). Tab state is independent from loop state. `buildCameraGrid()` skips DOM rebuild when ports are unchanged (prevents flicker on every status poll). `stopStream()` only clears `<img>` elements when on the stream tab, so Last Captures images persist when the loop stops. **Live Stream tab is currently disabled** (`disabled` attribute + `opacity:.35` CSS) because all camera cards share the same `/stream` URL (only one MUX channel visible at a time). Will be re-enabled once composite multi-camera stream is implemented. Default active tab is Last Captures.
- **GPIO event panel**: shows `TriggerEvent` list from the last inspection cycle.
  Each event displays: direction icon (→ output / ← input), BCM pin number, action type,
  result badge (OK = green, TIMEOUT = red, SENT = blue). Populated from `GET /api/last_result`.
  No hardware polling — reflects only what the software executed in the last cycle.
- Status polling every 2 s

### Endpoints (implemented)
| Method | Path | Description |
|---|---|---|
| GET | `/` | Home page |
| GET | `/setup` | Hardware wizard |
| GET | `/builder` | Sequence builder (requires sequence loaded) |
| GET | `/inspection` | Operational interface |
| GET | `/calibration` | Calibration page (capture → sweep → calibrate → deploy) |
| GET | `/stream` | MJPEG stream (lazy — only while loop is running) |
| GET | `/api/info` | Pi metadata (hostname, version, device_type, uptime, sequence count) |
| POST | `/api/validate_part_model` | Validate if part_model exists in saved sequences (excludes draft) |
| POST | `/api/setup/finalize` | Create draft from wizard with auto-generated paths, init hardware, redirect to /builder |
| POST | `/api/builder/capture_preview` | Capture frames from all channels. Returns `{captured, errors, hardware_wedged_message}` — `errors` is per-channel and never sticky (see section 23), `hardware_wedged_message` is a latched alarm (see section 23). |
| GET | `/api/builder/frame/<channel>` | Return JPEG for a channel |
| POST | `/api/builder/toggle_gpio` | `{pin, action: "turn_on"|"turn_off"}` — manual spotlight toggle from builder; uses `AppFactory._gpio` directly |
| POST | `/api/builder/update_pipeline` | `{camera_port, view, section, pipeline}` |
| POST | `/api/builder/update_hardware` | `{spotlight_gpio_pins, trigger_input_pin, camera_capture_resolution}` — updates draft hardware settings without camera topology |
| POST | `/api/builder/add_step` | `{step}` |
| POST | `/api/builder/remove_step` | `{step_number}` |
| POST | `/api/builder/validate` | Returns `{valid, errors}` |
| POST | `/api/builder/save_sequence` | Validate + save → reload factory → redirect to /inspection. In edit mode (`_edit_path` present in draft) overwrites the original file instead of creating a new numbered one. |
| GET | `/api/sequences` | List `config/sequence_*.json` |
| POST | `/api/load_sequence` | `{path}` → reload factory. Rejected (409) if running. |
| POST | `/api/load_sequence_for_edit` | `{path}` → copy sequence to draft, tag with `_edit_path`, init hardware, redirect to `/builder`. Rejected (409) if running. |
| POST | `/api/clone_sequence` | `{path, new_part_model}` → copy sequence's hardware/pipeline into a fresh draft with regenerated `paths` for `new_part_model` (tagged `_cloned_from`, not `_edit_path`), init hardware, redirect to `/builder`. Rejects duplicate `new_part_model` (409) and busy loop (409). |
| POST | `/api/resume_draft` | Resume the last auto-saved builder draft without going through setup wizard. Rejected (409) if running. |
| GET | `/api/status` | `{running, mode, dry_run, sequence, label, cycle_count, camera_ports, forced_scrap_counter, channel_status, hardware_wedged_message}` |
| POST | `/api/start` | Start loop |
| POST | `/api/stop` | Stop loop |
| POST | `/api/set_mode` | `{mode}` — rejected (409) if running |
| POST | `/api/set_dry_run` | `{dry_run: bool}` — rejected (409) if running |
| POST | `/api/set_label` | `{label}` — samples mode only |
| POST | `/api/set_forced_scrap` | `{count: int}` — forces next N cycles to route to NOK steps regardless of ML score; tags traceability as `OK_SCRAP`/`NOK_SCRAP` |
| GET | `/api/last_result` | Last Part as JSON (includes `triggers` list for GPIO event panel) |
| POST | `/api/capture_preview` | On-demand frame capture (loop idle only). Returns `{captured, errors, hardware_wedged_message}` — same shape as `/api/builder/capture_preview` (see section 23). |
| GET | `/api/frame/<channel>` | Return last captured JPEG. Priority: 1) manual preview (`IrisState`), 2) last inference frame from running controller (searches by channel suffix `view_name.split("_")[-1] == channel` when direct key lookup fails), 3) disk fallback from `images_path/latest/{view_name}.jpg` (written atomically by `GuiInferenceAdapter` after each cycle — persists across restarts), 4) 204. |
| GET | `/api/heatmap/<view>` | PNG error map heatmap |
| GET | `/api/calibration/status` | `{running, target, progress: {step, total, message, done, error}, best_blocks: {view_name: block}}` |
| POST | `/api/calibration/set_target` | `{target}` — `"train_ok"` \| `"test_ok"` \| `"test_nok"` |
| POST | `/api/calibration/start_capture` | Start GPIO-triggered capture loop for selected target |
| POST | `/api/calibration/stop_capture` | Stop capture loop |
| GET | `/api/calibration/image_counts` | `{train_ok: N, test_ok: N, test_nok: N}` per view |
| POST | `/api/calibration/run_sweep` | Start full block sweep (b3–b17) in background thread |
| POST | `/api/calibration/run_calibration` | Start final calibration for the selected block |
| GET | `/api/calibration/sweep_results` | Latest sweep results per view (block, AUC, Sep) |
| GET | `/api/calibration/last_frame/<view_name>` | Last captured image for a view served raw from disk (resized ≤ 800 px wide, no pipeline preprocessing). Query param `target=train_ok\|test_ok\|test_nok` |

### Streaming rules (implemented)
- **Lazy MJPEG**: generator runs only while `state.is_running`. Returns 204 otherwise.
- **FPS cap**: `time.sleep(1 / 5)` default; prevents network saturation.
- **JPEG quality**: 65 for stream, 80 for builder preview.
- **Connection counter**: tracked with `threading.Lock` for future multi-client limits.

### Mode switching
- Rejected (HTTP 409) while loop is active.
- Creates a new controller instance via `factory.create_inference_controller()` or
  `factory.create_samples_controller()`.

### Running Iris

**Development (local PC):**
```bash
# From InspectionApp/ root — must use -m so absolute imports resolve correctly:
python3 -m iris.IrisServer
# or via the Flask CLI:
flask --app iris.IrisServer:create_iris_app run --host 0.0.0.0 --port 5000
```

The `-m` flag is required because the codebase uses absolute imports (`app.src.…`).
Running `python3 iris/IrisServer.py` directly would add `iris/` to `sys.path` and
break all `app.src.*` imports.

**Production (Raspberry Pi) — gunicorn:**
```bash
# Install once:
pip install gunicorn

# Run manually from InspectionApp/ root:
gunicorn --config gunicorn.conf.py "iris.IrisServer:create_iris_app()"

# Or via systemd (recommended):
sudo cp setup/iris.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable iris
sudo systemctl start  iris
```

gunicorn uses `gthread` worker (1 worker, 4 threads) so the MJPEG stream and
API calls don't block each other. `gunicorn.conf.py` sets `accesslog=/dev/null`
to suppress the per-request HTTP log that would flood `TimestampedFileLogger`
with the browser's 3-second status-poll entries.

### Iris logs
`IrisServer.py` sets `logging.getLogger("werkzeug").setLevel(logging.ERROR)` at
import time. This suppresses Flask/werkzeug's per-request access log (the
`GET /api/status 200` lines) on both the dev server and gunicorn, so only
application-level `print()` / `[INFO]` / `[WARN]` lines appear in the log.
500-level errors are still logged by Flask via `app.logger`.

When started with `python3 -m iris.IrisServer` (dev), the `__main__` block wraps
the server in `TimestampedFileLogger`. With gunicorn, the `post_fork` hook in
`gunicorn.conf.py` starts `TimestampedFileLogger` inside the worker. In both
cases all `print()` output goes to:
- The terminal / systemd journal
- `{logs_path}/YYYYMM/YYYYMMDD_log-iris.txt`

`logs_path` is read from `config/default_values.json` (default `./data/logs/`).


## File tree (complete)

```
InspectionApp/
├── .github/
│   └── copilot-instructions.md    ← this file
├── app/
│   └── src/
│       ├── adapters/
│       │   ├── input/
│       │   │   ├── GuiInferenceAdapter.py
│       │   │   ├── GuiSamplesAdapter.py
│       │   │   └── GuiCalibrationAdapter.py
│       │   └── output/
│       │       ├── CoralUsbInferenceAdapter.py
│       │       ├── CpuInferenceAdapter.py
│       │       ├── CsiCameraAdapter.py
│       │       ├── LocalStorageAdapter.py
│       │       ├── NoOpMuxAdapter.py
│       │       ├── NullGpioAdapter.py
│       │       ├── PaDiMFeatureExtractorAdapter.py
│       │       ├── PaDiMInferenceAdapter.py
│       │       ├── RpiCsiMuxAdapter.py
│       │       ├── RpiGpioAdapter.py
│       │       └── UsbCameraAdapter.py
│       ├── core/
│       │   ├── models/
│       │   │   └── Part.py
│       │   ├── services/
│       │   │   ├── CalibrationService.py
│       │   │   ├── InspectionService.py
│       │   │   ├── SampleCaptureService.py
│       │   │   ├── SequenceBuilderService.py
│       │   │   └── SequenceExecutor.py
│       │   ├── settings/
│       │   │   └── SequenceSettings.py
│       │   └── utils/
│       │       ├── ImageProcessing.py
│       │       └── TimestampedFileLogger.py
│       ├── interfaces/
│       │   ├── ICamera.py
│       │   ├── IControllableCamera.py
│       │   ├── ICsiMux.py
│       │   ├── IGpio.py
│       │   ├── IInferenceEngine.py
│       │   ├── IPaDiMFeatureExtractor.py
│       │   └── IRepository.py
│       └── AppFactory.py
├── config/
│   ├── camera_catalog.json        ← camera model capability catalog (incl. resolution presets)
│   ├── counter.json               ← auto-generated, daily part ID counter
│   ├── default_values.json        ← hardware defaults (all sequences)
│   ├── device_catalog.json        ← device platform catalog (PC, RaspberryPi, Jetson)
│   ├── io_module_catalog.json     ← IO module catalog (v1/v2/v3)
│   ├── sequence_001.json          ← active product sequence (one file per product)
│   └── sequence_draft.json        ← auto-generated by setup wizard, used by builder
├── data/                          ← runtime-generated content
│   ├── images/                    ← captured images (train + inference + calibration)
│   ├── logs/                      ← timestamped log files
│   ├── models/
│   │   ├── backbones/             ← shared PaDiM ONNX backbones b3–b17 (pre-installed once)
│   │   └── {part_model}/          ← per-product: params.npz, thresholds.json, eval_config.json, graphs/
│   └── traceability/              ← inspection results (JSON Lines)
├── iris/
│   ├── __init__.py
│   ├── IrisServer.py              ← Flask app, create_iris_app(), all routes
│   ├── IrisState.py               ← Thread-safe state container
│   ├── templates/
│   │   ├── base.html              ← Shared layout
│   │   ├── home.html              ← /
│   │   ├── setup.html             ← /setup (3-step wizard)
│   │   ├── builder.html           ← /builder (canvas + pipeline)
│   │   ├── inspection.html        ← /inspection
│   │   └── calibration.html       ← /calibration (capture + sweep + calibrate)
│   └── static/
│       ├── css/iris.css
│       └── js/
│           ├── fabric.min.js      ← Fabric.js (download manually from fabricjs.com)
│           ├── canvas_tools.js    ← CanvasTools class (ROI, circle, rectangle, detection ROI)
│           ├── builder.js         ← Builder page logic
│           ├── inspection.js      ← Inspection page logic
│           └── calibration.js     ← Calibration page logic
├── gunicorn.conf.py               ← gunicorn config (workers, threads, accesslog=off, post_fork logger)
├── requirements.txt               ← Python dependencies (flask-cors, gunicorn, etc.)
├── scripts/
│   ├── export_backbones.py        ← standalone script to export MobileNetV2 backbone ONNX files
│   │                                (requires torch + torchvision; no Frambuesa dependency)
│   │                                Usage: python3 scripts/export_backbones.py [--blocks N …] [--force] [--verify]
│   └── cold_boot_camera_prewarm.py ← no-threads CSI camera prewarm (probe / channel <NAME>),
│                                     run by setup/run_camera_prewarm.sh before Iris starts (section 24)
├── setup/
│   ├── iris-browser.desktop       ← XDG autostart entry to open Chromium at boot
│   ├── iris.service               ← systemd unit for Iris web interface (gunicorn)
│   ├── iris_watchdog.service       ← systemd unit for the external liveness watchdog
│   ├── iris_watchdog.sh            ← watchdog script (polls /api/info, recovers a hung Iris)
│   ├── install_units.sh            ← installs iris.service/iris_watchdog.service, substituting user/home
│   ├── run_camera_prewarm.sh       ← runs cold_boot_camera_prewarm.py once per channel (own
│   │                                 process + external process-group-level kill each, section 24)
│   └── start_iris.sh              ← startup script: activates venv, starts gunicorn, opens Chromium
└── main.py
```

## Recent Architecture Updates & Bug Fixes (June 2026)

### 1. Hardware & MUX Synchronization (`RpiCsiMuxAdapter`)
- **I2C `-a` flag requirement:** The I2C multiplexer chip operates at address `0x00`. Linux's `i2cset` blocks addresses below `0x08` by default. The `i2c_cmd` in `default_values.json` must include the `-a` flag (e.g., `i2cset -y -a 10 0x00 0x04`).
- **Hardware Hot-Swap Stop-ChangeChannel-Start:** Switching CSI cameras physically via the MUX takes electrical time. `RpiCsiMuxAdapter.select_channel()` implements stop, change channel and start. Tactical `time.sleep()` is not required because the camera is stopped during the switch, allowing `libcamera` to release the bus and the relays to switch without electrical conflicts. Omitting this stop-change-start sequence causes `Error: Write failed` (I2C NACK) or `Camera frontend has timed out!` crashes.
- **GPIO State Parsing:** The `gpio_state` provided to the MUX from `default_values.json` is parsed as a dictionary with string keys representing BCM pins (e.g., `{"4": 0, "17": 0, "18": 1}`), not a list.

### 2. Camera Resolution Priority Chain
- `AppFactory._get_capture_resolution()` enforces a strict priority chain:
  1. Sequence/Draft JSON (`hardware.camera_capture_resolution`)
  2. `default_values.json`
  3. Absolute fallback (e.g., 4608x2592).
- This ensures that the Iris Builder Canvas and the backend capture hardware are perfectly synchronized, preventing ROI coordinates from scaling incorrectly during the "Update Preview" action.

### 3. OpenCV Color Space Corrections
- `Picamera2` outputs in RGB/YUV, but OpenCV's `cv2.imencode('.jpg', ...)` strictly expects **BGR** format.
- Conversions to `cv2.COLOR_RGB2BGR` (for main capture) and `cv2.COLOR_YUV2BGR_I420` (for lores MJPEG stream) are explicitly applied before encoding to prevent the "Smurf effect" (blue and red channels swapped in the UI).

### 4. Calibration & Samples Directory Structure (`view_name` vs `channel`)
- **Directory Pathing:** Images are saved and read using the full canonical `view_name` (e.g., `A1_A`) instead of the physical `camera_port` (`A`).
- **SampleCaptureService:** Modified to parse `sequence_steps` (via `camera_action`) to derive the exact `view_name` for saving training images.
- **AppFactory & IrisServer:** `AppFactory.get_calibration_image_dirs()` and `IrisServer.api_calibration_last_frame()` strictly use `view_name` to construct paths (`images_path/test/OK/{view_name}`). This prevents `insufficient test images — skipped` errors during the PaDiM sweep.

### 5. Multi-Environment Startup (`start_iris.sh`)
- The startup script uses an array iteration fallback to locate and activate the correct Python virtual environment across different deployment machines (`visredPC`, `visred`, `mezt`) before starting Gunicorn.

### 6. Strict Preprocessing Pipeline Mapping (Frontend & Backend)
- **Frontend (`builder.js`):** Forces pipeline `view` keys to match strict section IDs (e.g., `section_1`, `section_2`) instead of allowing arbitrary or generic names (`section_view`). This ensures a 1:1 mapping between the UI tabs and the JSON structure.
- **Backend (`SequenceSettings.py`):** Uses exact `view_name` lookup (e.g., `section_1_A`) without redundant string concatenations. This prevents ROI collisions where a pipeline for `section_2` would overwrite `section_1` just because they share the same camera port.
- **Calibration (`IrisServer.py`):** `_build_view_configs` extracts ROIs matching exact `view_name` first, using the camera port solely as a fallback.

### 7. RAM/Disk State Synchronization (Post-Calibration)
- **The Issue:** `CalibrationService` writes new `thresholds.json` and `eval_config.json` to disk, but `AppFactory` (and `SequenceSettings`) held stale versions in RAM, causing `KeyError` on the next inference cycle.
- **The Fix:** `IrisServer.py` now automatically invokes `_load_sequence()` inside the `_on_calibration_done` callback. This forces the inference engine to reload the newly calibrated parameters from disk to RAM without requiring a manual server restart or a "Load" button click.

### 8. 6-Digit Part ID Expansion (`GuiInferenceAdapter`)
- **The Issue:** The previous 4-digit formatting (`:04d`) capped unique sequential daily part IDs at 9,999. High-throughput production lines can exceed 10,000 parts per day, causing formatting and indexing collisions.
- **The Fix:** Modified `GuiInferenceAdapter._generate_part_id()` to pad the daily counter with 6 digits (`:06d`). The generated string format is now strictly `YYYYMMDD-NNNNNN` (e.g., `20260609-000042`), safely scaling up to 999,999 daily inspection cycles.

### 9. Manual Spotlight Toggles & Safety (Builder)
- **The Feature:** Added a "Hardware Control" panel in the Builder UI to manually toggle spotlights on/off while configuring ROIs, improving visibility for setup.
- **Backend (`IrisServer.py`):** Added a new `POST /api/builder/toggle_gpio` endpoint that uses `AppFactory._gpio` to execute `turn_on` / `turn_off` commands on demand without triggering a full inspection cycle.
- **Frontend (`builder.js` & `builder.html`):** Dynamically renders toggle buttons based on `hardware.spotlight_gpio_pins` from the draft. Implemented a Flexbox grid to sit alongside the canvas tools cleanly.
- **Safety Mechanism (`keepalive`):** Added a `window.addEventListener("beforeunload")` hook in `builder.js`. If the user navigates away, reloads, or closes the tab while spotlights are ON, the browser fires an emergency `toggle_gpio` OFF request using `keepalive: true`, preventing hardware burnout.

### 10. Per-View Exposure Control & Camera Buffer Sync
- **The Feature:** Allow exposure and lens position adjustments per view in the Builder and reflect them accurately in the live preview.
- **Frontend (`builder.js`):** `btn-capture` now sends `{ section: activeSectionId }` to `/api/builder/capture_preview` so the backend knows which pipeline to read. Fixed pipeline array duplication where tools were unshifted without clearing old ones.
- **Backend (`IrisServer.py`):** `/api/builder/capture_preview` reads the specific section's pipeline from `state.draft`, extracts `set_time_exposure` and `set_lens_position`, and passes them to `AppFactory.capture_preview_frame()`.
- **Hardware/Buffer Fix (`CsiCameraAdapter.py`):** `Picamera2` processes controls asynchronously and queues frames in an internal buffer. To prevent returning "stale frames" with old lighting, `set_exposure_time` and `set_lens_position` now explicitly `stop()` the camera, apply controls, and `start()` it again, guaranteeing the very next captured frame uses the updated settings.

### 11. Forced SCRAP Mode & Dry Run Traceability
- **The Feature:** Added a UI toggle that forces the next `N` inspection cycles to route to NOK steps, overriding the ML score, and tags the traceability logs for both SCRAP and Dry Run modes.
- **Domain (`Part.py`):** Added `forced_scrap: bool` and `dry_run: bool` properties.
- **Backend (`SequenceExecutor.py` & `GuiInferenceAdapter.py`):** The adapter maintains a `_forced_scrap_counter`. If `> 0`, it flags the `Part` and `SequenceExecutor` forces the routing to execute NOK steps (`step_number < 0`).
- **Traceability Integration:** The `overall_status` in `LocalStorageAdapter`, `GuiInferenceAdapter._serialize_result`, and `SequenceExecutor` logs dynamically appends `_SCRAP` (if forced) or `_DR` (if dry run) to the ML result (e.g., `OK_SCRAP`). This ensures the database, the terminal, and the web UI all reflect the true operational context identically.
- **Frontend (`inspection.js` & `IrisServer.py`):** Added `POST /api/set_forced_scrap` endpoint. The UI button uses an IIFE-scoped `apiPost` to toggle the mode and updates its label with the remaining cycles via the `/api/status` payload.

### 12. Auto-Resize to ROI Dimensions (Builder QoL)
- **The Feature:** When the operator applies drawn shapes to the pipeline, the system automatically detects if an `apply_roi_crop` tool is present. If so, it matches the `resize_to_training_resolution` dimensions to the exact width and height of the ROI, preventing aspect ratio distortion.
- **Frontend (`builder.js`):** Intercepted the `btn-apply-pipeline` logic to search the exported `canvasPipe` array for an ROI. If found, it dynamically overrides the `targetW` and `targetH` variables and updates the UI inputs. If no ROI is found, it falls back to the user inputs or defaults to 525x525.

### 13. Traceability Synchronization & Empty Nest Logging Fix
- **The Issue:** `LocalStorageAdapter.save_inspection_result()` was originally called inside `InspectionService.execute_full_inspection_from_frames()`. Because inference occurs at step 1001, parts were saved *before* NOK steps executed and *before* `duration_s` stopped, resulting in `null` times and missing GPIO trigger logs. Additionally, empty nests (`piece_detected=False`) skipped inference entirely and were never logged.
- **The Fix:** Decoupled persistence from inference. Removed save calls from `execute_full_inspection_from_frames` and created `InspectionService.save_results()`. `SequenceExecutor` now explicitly calls this method at the very end of `run()`, guaranteeing that `duration_s`, all NOK GPIO triggers, and empty nest evaluations are perfectly preserved in the JSONL logs.

### 14. Builder Hardware Settings Modal
- **The Feature:** A dedicated modal within the sequence builder that allows operators to safely edit electrical parameters (Spotlight GPIO pins, Trigger Input pin) and Capture Resolution on the fly without running through the setup wizard again.
- **Security:** The modal intentionally excludes camera topology (camera model, type, and counts) to prevent operators from inadvertently corrupting the neural network alignment or the sequence step logic.
- **Frontend (`builder.html`, `builder.js`):** Extracts and renders data dynamically from `draft.hardware`. On save, updates the local variables and automatically forces a re-render of the "Hardware Control" UI panel in case spotlight pins were added or removed.
- **Backend (`IrisServer.py`):** `POST /api/builder/update_hardware` receives the targeted payload, injects it into `draft["hardware"]`, and persists the JSON.

### 15. PLC Independence & Synchronized Sample Capture
- **The Issue:** `SampleCaptureService` was ignoring intermediate `wait_for_input` GPIO steps, breaking the robot handshake. Additionally, inference `ValueError`s crashed the thread entirely, which could cause erratic behavior or send unexpected signals to the PLC.
- **Sample Capture Fix (`SampleCaptureService.py`):** Rewrote `run_capture_cycle()` to explicitly iterate over `self._sequence_steps` (0 to 999), executing `gpio_action` and `camera_action` in strict order to maintain the PLC handshake. `wait_for_trigger()` is intentionally a no-op (returns `True` immediately) — the real PLC trigger wait is handled by the first `wait_for_input` `gpio_action` step inside `run_capture_cycle()` using the `expected_value` and `timeout` defined in the sequence JSON. Keeping `wait_for_trigger()` as a no-op avoids a double-wait on the same pin.
- **Inference Shield (`SequenceExecutor.py`):** Wrapped the execution loop in a `try...except`. Following the "PLC as master" philosophy, if inference fails, the vision system flags `part.system_error_paused = True` and **explicitly skips NOK steps**. This allows the PLC to handle the lack of response via its own logic without receiving rogue reject signals.
- **Auto-Pause (`GuiInferenceAdapter.py`):** The `_run_loop` detects `part.system_error_paused` and calls `self.pause()` automatically. The machine safely halts its own inspection loops, preventing a catastrophic cascade of errors.

### 16. CSI Camera Sensor Stabilization After MUX Channel Switch
- **The Issue:** `start_stream()` in `CsiCameraAdapter` had no delay after `picam.start()`. After a MUX channel switch (`stop_stream()` → GPIO+I2C → `start_stream()`), the first captured frame could be from the transition rather than the stable new channel.
- **Fix (`CsiCameraAdapter.start_stream()`):** Added `time.sleep(0.05)` after `picam.start()`, matching the delay already present in `initialize_camera()`.
- **Update (confirmed Sept 2026):** `0.05` s was not always enough once 2+ cameras are actually switched (see item 21). Tuned upward (~0.08–0.1 s) until channel C stopped reproducing `Camera frontend has timed out!` — check the live value in `start_stream()` before assuming it is still `0.05`. There is no official spec value for this: physically multiplexing a MIPI CSI-2 link between different sensors is a board-level workaround libcamera/Picamera2 have no built-in awareness of, so the delay is empirical, not documented anywhere.

### 17. libcamera "Configured State" Bug on Re-initialization
- **The Issue:** When `_load_sequence()` is called (e.g. loading a sequence for editing) after the inference loop has been running, `initialize_camera()` fails with `Camera in Configured state trying acquire() requiring state Available`. This happens because:
  1. `close_camera()` / `release_camera()` had `stop()` conditioned on `_is_initialized`. If the MUX stopped the stream just before shutdown, `_is_initialized=False` and `stop()` was skipped, leaving libcamera in "Configured" state instead of "Available".
  2. If `Picamera2()` raises during construction (before being assigned to `self.picam_instance`), the partially-acquired instance was never cleaned up, cascading into all subsequent retries.
  3. The between-retry sleep was only 20ms — libcamera needs ~1–2 seconds to transition from "Configured" to "Available".
- **Fix (`CsiCameraAdapter`):**
  - `close_camera()` and `release_camera()`: removed the `if self._is_initialized:` guard on `stop()` — always attempt `stop()` before `close()` regardless of state, wrapped in `try/except`.
  - `close_camera()` and `release_camera()`: increased sleep from 100ms to 300ms.
  - `initialize_camera()`: introduced local variable `new_cam` to hold each attempt's `Picamera2()` instance; if construction raises before assignment to `self.picam_instance`, `new_cam.close()` is explicitly called to release the partial acquisition.
  - `initialize_camera()`: increased post-`release_camera()` sleep from 200ms to 500ms.
  - `initialize_camera()`: increased between-retry sleep from 20ms to 1500ms.

### 18. GPIO Thread-Zombie Race Condition (`stop_loop` → `interrupt` chain)
- **The Issue:** `wait_for_input(timeout_ms=0)` was an unconditional `while True: button.is_pressed` loop with no way to exit short of a hardware signal. `stop_loop()` called `thread.join(timeout=15)` which expired after 15 seconds while the thread was still alive. Any subsequent Stop→Edit, Stop→Start, or Stop→mode-change within that 15-second window resulted in:
  1. `gpio.close()` called while thread still polling → `GPIODeviceClosed` crash.
  2. Multiple threads spawned simultaneously, each calling `Picamera2()` → `Camera __init__ sequence did not complete` (libcamera does not support concurrent instances).
  These two symptoms produced the log pattern of multiple `[ERROR] Failed to initialize camera (attempt 1/5)` entries with identical millisecond timestamps and interleaved tracebacks.
- **Root cause confirmation:** Gaps of exactly 15 seconds between `[OK] GuiInferenceAdapter: inspection loop started` and `[OK] CSI camera closed` in the log proved the `join(timeout=15)` was expiring without the thread having exited.
- **Fix — `interrupt()` / `reset_interrupt()` chain across 7 files:**
  - **`IGpio` interface**: two new abstract methods `interrupt()` (signal any blocking `wait_for_input` to return `False` immediately) and `reset_interrupt()` (clear the flag before the next `start_loop()`).
  - **`RpiGpioAdapter`**: adds `_interrupted: threading.Event` (separate from `_closed`). `interrupt()` sets it; `reset_interrupt()` clears it; `close()` sets both. `wait_for_input(timeout=0)` uses `while not self._closed.is_set() and not self._interrupted.is_set()` — exits within the next 10 ms poll when either is set. Finite-timeout variant also checks both events each iteration.
  - **`NullGpioAdapter`**: `interrupt()` and `reset_interrupt()` are no-ops (never blocks).
  - **`SequenceExecutor`**: `interrupt()` and `reset_interrupt()` delegate to `self._gpio`.
  - **`SampleCaptureService`**: same delegation.
  - **`GuiInferenceAdapter.stop_loop()`**: calls `self._executor.interrupt()` before `join()` → thread exits in <10 ms instead of 15 s.
  - **`GuiInferenceAdapter.start_loop()`**: calls `self._executor.reset_interrupt()` before spawning the thread — clears the flag so the first `wait_for_input` of the new cycle works normally.
  - **`GuiSamplesAdapter`**: same pattern via `self._service.interrupt()` / `self._service.reset_interrupt()`.
- **Why two events** (`_closed` vs `_interrupted`): `_closed` is permanent (set by `close()`, never cleared — hardware is gone). `_interrupted` is temporary (set by `stop_loop()`, cleared by `start_loop()`). Using a single event would cause Stop→Start (without reload) to short-circuit `wait_for_input` on the very first call of the new cycle.

### 19. PaDiM Shape Mismatch at Inference (`operands could not be broadcast`)
- **The Issue:** `ValueError: operands could not be broadcast together with shapes (30,31,100) (32,35,100)` in `PaDiMInferenceAdapter.predict()` at `diff = (feat - mean)`.
- **Cause:** `feat` is the backbone output for the current frame; `mean` was stored during calibration. The backbone produces feature maps at ~1/16 of the input spatial resolution, so a mismatch of `(30,31)` vs `(32,35)` implies ~30–48 px of difference in the network input. This happens when the `resize_to_training_resolution` or ROI parameters for a view are changed in the sequence JSON **after** calibration without re-running calibration.

### 20. Reference Image for Offline ROI Setup (Builder)
- **The Feature:** Allows an operator to draw ROIs and pipeline shapes in the Builder without the production line running (e.g. after a sequence was accidentally deleted). A local image file is loaded as the canvas background for the currently selected camera, replacing the live camera preview.
- **Frontend (`builder.html`):** Added `📂 Reference image` button (`id="btn-load-ref"`) and a hidden `<input type="file" id="ref-img-input" accept="image/*">` in the capture toolbar, next to the existing `📷 Update Preview` button.
- **Frontend (`builder.js`):** IIFE that binds `btn-load-ref` → triggers `ref-img-input.click()` for the active channel. On file selection, calls `canvasMap[activeChannel].setBackground(file)` — the `Blob` overload of `setBackground` already existed in `canvas_tools.js` (accepts `Blob | string`). No backend changes required.
- **Multi-camera & multi-section behaviour:** The button loads the image for whichever camera card is currently selected (`activeChannel`). For 1–4 cameras, the operator clicks each card and loads its reference image separately. The background persists when switching sections (`setActiveSection` calls `clearObjects()` which removes shapes but not the Fabric.js background image), so one load per camera covers all sections.
- **Use case:** Sequence deleted, line stopped → open Builder → create new sequence → click camera card → 📂 Reference image → pick any image from `data/images/<part>/train/OK/<view>/` → draw ROI as normal → save sequence.
- **Fix:** Re-calibrate the affected views with the current pipeline, or restore `resize_to_training_resolution` and ROI coordinates to the values used during the original calibration run. The expected spatial dimensions can be inferred from `padim_*_params.npz`: `mean.shape[:2]` gives the expected `(H', W')` of the feature map.

### 21. Root Cause of Intermittent CSI MUX Freezes: Sensor Modes, Resolution & Real FPS Limits (Sept 2026)
- **Symptom:** With only 1 CSI camera configured (no MUX switching ever happens) the system is 100% stable. From 2 cameras onward (real `select_channel()` switches start happening) `capture_frame` intermittently fails with `Camera frontend has timed out!`, historically observed on channel C. The legacy "Frambuesa" desktop app (`src/`) never shows this on the same physical hardware — traced to two compounding, independently confirmed factors, not a cable/MUX-chip defect (both were swapped/checked with no change):
  1. **Capture resolution.** Legacy `ajustes_generales.json` captures at `1280×720`; Iris defaulted to the sensor's full native resolution (`4608×2592` for imx708 / `4056×3040` for imx477) — far more CSI/ISP bandwidth per frame, more likely to tip a marginal channel into a real ISP stall.
  2. **Post-switch settle delay.** See items 1 and 16 — physically switching which sensor is electrically connected to the CSI-2 lanes (MUX) forces a real re-sync of the differential link; a fixed delay after `start_stream()` that is too short races against that re-sync and fails intermittently. With 1 camera this code path never executes, hence never fails.
- **imx708 / imx708_wide real sensor modes** (from `rpicam-hello --list-cameras`, confirmed with a physical unit) — only 3 modes exist, **not a continuum**:
  | Mode | Resolution | HW max fps | Field of view |
  |---|---|---|---|
  | Crop | 1536×864 | 120.13 | **Cropped** (`(768,432)/3072x1728` \u2014 loses ~1/3 width & height) |
  | Binned 2×2 | 2304×1296 | 56.03 | Full (`(0,0)/4608x2592`) |
  | Native | 4608×2592 | 14.35 | Full (`(0,0)/4608x2592`) |
  Only the binned and native modes preserve full FoV. There is **no intermediate hardware mode** between them.
- **imx477 (HQ Camera / "RPIcam Lens") real sensor modes:** same pattern — full-FoV crop rectangle is `(0,0)/4056x3040`, matched only by `4056×3040` (native, 14.0 fps) and `2028×1520` (binned 2×2, 53.8 fps). `1332×990`, `2028×1080`, `4056×2160` are all genuine crops (reduced FoV). **`camera_catalog.json`'s `low` preset for `imx477` (`1012×760`) does not match any real sensor mode** \u2014 it falls back to the smallest available crop mode, silently narrowing the FoV relative to `high`/`medium`. Not currently fixed; flag if `low` is ever selected for this camera model.
- **Empirically confirmed: requesting an "in-between" resolution gives zero relief.** Tested `3456×1944` (75% of `4608×2592`, same 16:9 aspect ratio, so no crop) on real hardware via `rpicam-vid`: the pipeline log shows `configuring streams: (1) 4608x2592-SBGGR10_CSI2P/RAW` i.e. it still selects the **native full-resolution raw sensor mode** and only downscales the output in the ISP. Measured fps was ~14.6, identical to running at full `4608×2592` \u2014 confirms there is no CSI-bandwidth benefit from arbitrary output sizes; only requesting an output size ≤ a real binned mode's size actually engages that lighter mode.
- **Gotcha: video configurations silently cap fps near 30 regardless of the sensor mode's HW max.** Measured with `rpicam-vid` (no `--framerate` override): the binned `2304×1296` mode (56.03 fps HW capability) only achieved **30.24 fps**; the native `4608×2592` mode (14.35 fps HW capability, below the ~30fps default) achieved its real ~14.6 fps unimpeded. This means `Picamera2.create_video_configuration()` (used by `CsiCameraAdapter.initialize_camera()`) likely applies the same implicit default frame-duration target \u2014 simply switching `capture_resolution` to the binned mode will **not** automatically unlock its true 56 fps unless `FrameDurationLimits` is also explicitly relaxed in `init_controls`. Verify before assuming a resolution change alone buys back the full theoretical speed.
- **Diagnostic commands developed this session** (must stop Iris/gunicorn first \u2014 `Picamera2` cannot be opened twice):
  - Manually select a MUX channel without the app (values from `default_values.json` → `csi_channels`), e.g. channel A: `raspi-gpio set 4 op dl; raspi-gpio set 17 op dl; raspi-gpio set 18 op dh; i2cset -y 10 0x70 0x00 0x04`.
  - Live FPS overlay (needs an attached display, not pure SSH): `rpicam-hello -t 0 --width <W> --height <H> --info-text "%fps fps"`.
  - Real measured FPS, independent of the monitor's refresh rate (works headless over SSH): `rpicam-vid -t 5000 --width <W> --height <H> --nopreview --codec yuv420 -o /dev/null --save-pts /tmp/pts.txt` then `awk 'NR==1{first=$1} {last=$1; n=NR} END{printf "%d frames, %.2f fps\\n", n-1, (n-1)/((last-first)/1000)}' /tmp/pts.txt`.
  - To rule out auto-exposure as the fps limiter, force a short manual shutter: add `--shutter 5000 --gain 1` to either command above.
- **Recommendation for 2+ camera setups:** prefer the binned full-FoV mode (`2304×1296` for imx708, `2028×1520` for imx477) over the native resolution \u2014 same FoV, ~4× less raw data over CSI, and a much faster HW readout ceiling, giving more timing margin for the post-switch settle delay. Confirm the real achieved fps with the commands above (not just the theoretical HW max) after any resolution change.

### 22. `recover_hardware()` Race Condition: `'NoneType' object is not subscriptable` in `start_stream()` (Sept 2026)
- **Symptom, confirmed from real production logs (not synthetic tests):** intermittently, `start_stream()`/`capture_frame()` raises `'NoneType' object is not subscriptable` from inside Picamera2 itself, always shortly after a `close()` call, with no `initialize_camera()` in between. Once it happens, every subsequent capture fails with `Camera is not initialized. Call initialize_camera() first.` until something re-initializes the camera. **Independent of resolution and of the post-switch settle delay** (section 21) \u2014 reproduces at any resolution/delay combination, including configs previously reported as "stable".
- **Root cause:** `SequenceExecutor.recover_hardware()` (called from `GuiInferenceAdapter._run_loop()`'s crash-recovery block after a capture timeout) used to call `self._camera.close_camera()` and `self._camera.initialize_camera()` as two **separate** `hw_lock` acquisitions:
  ```python
  self._camera.close_camera()       # acquires + releases hw_lock
  self._camera.initialize_camera()  # acquires + releases hw_lock separately
  ```
  Between the two calls there is a window where `hw_lock` is free. If another thread (e.g. a Flask request thread running `api_builder_capture_preview()`, which loops over every camera channel calling `capture_preview_frame()` \u2192 `select_channel()`/`capture_frame()`) acquires the lock in that window, it runs `start_stream()` against the `picam_instance` that was *just closed*, and Picamera2's internal state (already torn down by `close()`) raises `'NoneType' object is not subscriptable'`.
  - **Fix applied:** wrap `close_camera()` + `initialize_camera()` in a single `with self._camera.hw_lock:` block in `SequenceExecutor.recover_hardware()` (and in the equivalent legacy fallback path in `GuiInferenceAdapter._run_loop()`, gated by `_USE_FULL_HARDWARE_RECOVERY`). `hw_lock` is an `RLock`, so the nested `with self.hw_lock:` inside `close_camera()`/`initialize_camera()` remains safe (reentrant).
- **Why this looked "random" / "hardware-cache-like" / tied to switching sequences:** this is a textbook race condition (a "Heisenbug"). Its trigger depends on the exact timing overlap between the background inspection loop's crash-recovery and any concurrent camera-using request (typically a Builder preview click) \u2014 a window of a few milliseconds. Changing the active sequence (different camera count \u2192 different per-cycle timing) shifts *when* the recovery thread runs relative to user clicks, changing the odds of a collision, without touching the actual bug. **The absence of the symptom in a given session does NOT mean it's fixed** \u2014 only that the race window wasn't hit that time. Do not use "it stopped happening" as evidence a hardware-tuning change (resolution, delay) solved this; only the atomic-lock fix addresses the actual cause.
- **Rejected mitigation \u2014 "ghost/warm-up capture on sequence load":** doing one throwaway capture at sequence-load time with a longer (100\u2013200 ms) settle delay was proposed as a fix. Rejected: it can only reduce the odds of hitting the race window near startup, but `recover_hardware()` runs at arbitrary times during production (any time a capture times out), so the unguarded window persists indefinitely regardless of any startup warm-up. Only closing the window itself (atomic lock, applied above) removes the bug.

### 23. Per-Channel Camera Isolation, Production Failure Attribution & Hardware-Wedged Alarm (Sept 2026)
- **The Issue:** A physically disconnected camera on one CSI channel would stall the single shared `Picamera2` instance for **all** channels, not just the disconnected one (same root cause as section 21/22 \u2014 one shared instance, MUX-switched). In the Builder/Inspection preview flow this meant: (1) the operator saw the failure reported for whichever channel happened to be captured next in the loop (e.g. B, then C, then D) rather than the physically disconnected one, giving the false impression of a channel-mislabeling bug; (2) once the shared instance stalled hard enough that `initialize_camera()` itself could no longer create a working `Picamera2` object (confirmed in production logs: `Camera in Running state trying acquire() requiring state Available`, repeating on every subsequent attempt), automatic recovery retried forever and could never actually succeed, since the underlying libcamera/native resource was never truly released (the timeout-guard's fire-and-forget daemon thread can abandon the *Python* reference but not cancel the blocked *native* call) \u2014 this is a hard architectural limit, not a bug, and it once escalated into a gunicorn worker crash. In production cycles (`SequenceExecutor.run()`), a camera failure was already correctly aborting the cycle (`part.system_error_paused = True`, confirmed to already block inference \u2014 Step 1/Step 2 share one `try` block in `run()`), but the channel that actually failed was never preserved anywhere, so the operator had no way to tell which physical camera to check.
- **Channel attribution is correct, not text-parsed:** every place that reports "which channel failed" \u2014 `AppFactory._channel_status[channel]`, `Part.failed_channel` \u2014 is keyed directly off the real loop/parameter variable (`channel`/`camera_port`) at the exact call site, never inferred from parsed log/error text. This makes the whole feature immune to the channel-mislabeling class of bug; an apparent "wrong channel" report always means a real cascade (see above), not an attribution bug.
- **Fix \u2014 per-channel isolation (Builder + Inspection preview, both call `AppFactory.capture_preview_frame()`):**
  - `AppFactory._channel_status: dict[str, str | None]` \u2014 refreshed on every `capture_preview_frame()` call, never sticky (a channel simply reflects its most recent attempt; no manual "recheck" button). Exposed via `GET /api/status` (`channel_status`) and rendered as a `.camera-card.disconnected` badge ("Check camera") on the matching card in both `builder.js` and `inspection.js`.
  - `AppFactory._recover_camera_after_failure()`: best-effort, never raises \u2014 `time.sleep(1.5)` then `close_camera()` (own exception swallowed so `initialize_camera()` is still attempted) \u2192 `initialize_camera()` \u2192 `mux.reinitialize()`.
  - `CsiCameraAdapter._PICAM_CALL_TIMEOUT_S` lowered `8.0 s \u2192 1.0 s` (applies globally, including production's `recover_hardware()`) \u2014 accepted trade-off: faster failure detection and much shorter worst-case stall (~24 s \u2192 ~3 s), at the cost of a higher false-positive risk on a legitimately slow-but-healthy camera under load/thermal throttling.
  - `close_camera()`'s two `except Exception:` blocks (`stop()`/`close()`) now abandon the instance (`picam_instance = None`, `_is_initialized = False`) on timeout, matching `stop_stream()`/`start_stream()`'s existing pattern (previously only those two did this, letting `close_camera()`'s own timeout block `initialize_camera()` from ever being attempted in the same recovery call).
- **Fix \u2014 production channel attribution:** `Part.failed_channel: str | None` / `Part.failed_channel_error: str | None` (dynamic attributes, same pattern as `system_error_paused` \u2014 not part of the dataclass). Set by `SequenceExecutor._execute_camera_action()`, `_execute_detect_piece_action()`, and `_execute_wait_for_piece_action()` in a `try/except` around their `select_channel()`/`capture_frame()` calls \u2014 the exception is still re-raised unchanged, so the existing abort behavior (`ERROR_ABORTED`, inference already skipped) is untouched; this is purely additive metadata. Exposed via `GuiInferenceAdapter._serialize_result()` \u2192 `GET /api/last_result` \u2192 `inspection.js` `renderResult()` shows "\u26a0 Camera X failed: ..." next to the `ERROR_ABORTED` badge and marks the matching camera card.
- **Fix \u2014 hardware-wedged latched alarm (no auto-restart):** when `_recover_camera_after_failure()`'s `initialize_camera()` call itself fails, `AppFactory._hardware_wedged_message` is latched to a fixed operator-facing message and **never auto-clears** \u2014 `capture_preview_frame()` fails fast for every channel while it is set, instead of piling up more orphaned recovery threads. Exposed via `channel_status`'s sibling field `hardware_wedged_message` (`GET /api/status` + both preview endpoints' JSON) and rendered as a persistent critical banner (`showCriticalBanner()`, reused/added in `inspection.js`/`builder.js`) telling the operator to use the "\u23fa Restart Iris" topbar button (see the "Reboot button" row in Pending work) \u2014 **deliberately not an automatic restart**. Rationale (ISA-18.2/ISA-101 alarm management): equipment faults must surface as a persistent, operator-acknowledged alarm, never a silent auto-recovery \u2014 an automatic restart would hide the diagnostic evidence from the operator and risks a restart-loop if the camera is genuinely still disconnected. `GuiInferenceAdapter._stopped_reason`'s existing production message (consecutive-failures case) was also corrected: it previously said "restart the inspection loop", which is misleading once a channel is truly wedged at the OS level \u2014 Start/Stop does not release the native resource, only a real "Restart Iris" does. `_hardware_wedged_message` and `_stopped_reason` are separate mechanisms (preview vs. production recovery paths, respectively) but now use consistent wording.

### 24. Cold-Boot-Only Camera Prewarm via a Rustic Single-Process Script ("videobuf2 wedge") (Sept 2026)
- **The Issue:** On a genuine cold boot (Pi physically powered on), the first `select_channel()`/`start_stream()` on some CSI channels reproducibly fails with `Camera frontend has timed out!` / `Dequeue timer expired`, and the kernel log (`dmesg`) shows `videobuf2_common: driver bug: stop_streaming operation is leaving buffer 0 in active state` \u2014 a kernel-level driver bug, not an application bug. Confirmed to affect **both** Iris and the legacy "Frambuesa" desktop app equally on cold boot with 2+ cameras (an earlier hypothesis that legacy "never fails" was tested on real hardware and falsified). Once wedged this way, only killing the whole process (closing the fd at the kernel level) clears it \u2014 no in-process recovery (MUX reinit, camera close/reinit, retrying `initialize_camera()` in the same process) can release it, because the kernel resource itself is stuck, not just the Python/libcamera object graph. Confirmed on real hardware (`20260908_log-iris.txt`) that an in-process retry loop across channels (an earlier attempt, `AppFactory._warmup_all_channels_cold_boot()`) does **not** work: channel A/B initialize, B's capture then times out, and every subsequent channel's `initialize_camera()` fails all its retries \u2014 that in-process approach was reverted in favor of the fix below.
- **Root cause narrowed to cold boot specifically:** a plain process/service restart (`systemctl restart iris`, "Restart Iris" button) \u2014 same hardware, same kernel, no power cycle \u2014 does **not** reproduce the wedge. This means the fragile step is specifically the **very first** stream start per physical channel after a power-on, not every channel switch. Subsequent switches on an already-warmed-up channel are reliably fine.
- **Two disposable-subprocess designs tried and reverted (reported erratic in the field):** first a per-channel-in-isolation subprocess (one subprocess per single channel), then a cumulative-prefix version (subprocess *i* re-walks channels `[0:i]` on the same running stream, always restarting from 'A'). Both spawned throwaway `python scripts/cold_boot_camera_prewarm.py` child processes built on `CsiCameraAdapter`/`RpiCsiMuxAdapter`. Field testing showed inconsistent results (channels initializing but then failing capture with the same 1 s timeout) and the user reported the overall behavior as erratic \u2014 both designs were abandoned in favor of the rustic single-process fix below.
- **Fix (superseded twice \u2014 see below for the current design) \u2014 rustic single-process prewarm:** first rewritten to mirror the proven-in-the-field legacy app (`src/01-MuestrasPi_InferenciaPi`) as literally as possible instead of reusing Iris's own adapters: raw `gpiozero.OutputDevice` pins, raw `os.system(i2c_cmd)`, a single bare `Picamera2()` instance, plain `stop()`/`start()` calls around each channel switch. A field-tested hang (bare `capture_array()` blocking forever once the ISP wedged, confirmed real on `bash setup/start_iris.sh`) led to a thread-based call timeout (`_call_with_timeout()`, a `threading.Thread`+`Event` wrapper duplicated locally, same `1.0s` threshold as `CsiCameraAdapter._call_picam_with_timeout`) being added around every `stop()`/`start()`/`capture_array()` call, plus a 5-attempt bash retry loop in both boot paths (bumped from 3 after empirical testing showed each fresh-process attempt only clears one additional channel \u2014 attempt *N* reliably got through the first *N* channels before wedging on the next).
  - **Both of those fixes were themselves abandoned (Sept 9 2026), for two reasons confirmed on real hardware:**
    1. **The thread-based timeout itself, not just the hang, correlated with erratic/corrupted preview images on channel B** \u2014 the user observed a wrong image being returned, not just a slow one, strongly suggesting the abandoned-daemon-thread-may-still-touch-the-instance race described in the code's own docstring was a real, active problem, not just a theoretical one.
    2. **A channel warmed successfully by the prewarm script did not reliably stay un-wedged for a different, later process.** Iris's own process wedged on channel 'C' (`builder_preview_C`) seconds after startup, even though the prewarm script's own last attempt had just reported that same channel 'C' as `[OK]` in a separate process moments earlier.
  - See git history / prior revisions of this file for the exact abandoned code if this pattern resurfaces and needs re-examining.
- **Current fix \u2014 per-channel, no-threads prewarm with an external, process-group-level kill (`scripts/cold_boot_camera_prewarm.py` + `setup/run_camera_prewarm.sh`, Sept 9 2026, revised same day):** removes ALL threading from the Python script (100% bare/blocking calls again, exactly like legacy) and moves timeout enforcement entirely OUTSIDE the process, to a new shared shell orchestrator, `setup/run_camera_prewarm.sh`, run by both boot paths.
  - **`scripts/cold_boot_camera_prewarm.py`** takes a required CLI arg, `probe` or `channel <NAME>`, and has no notion of an in-process timeout at all \u2014 if a call hangs, the process just hangs, and it is the caller's job to kill it.
    - `probe` (sacrificial): starts the camera and takes exactly **one** throwaway `capture_array("lores")` \u2014 no GPIO/I2C, no channel switch at all (the MUX is left wherever it physically is). Purpose is only to absorb the very-first-stream-after-boot wedge in a disposable process.
    - `channel <NAME>` (e.g. `channel A`): warms up exactly **one** physical channel in its own fresh process \u2014 starts the camera on whatever channel the MUX is currently on, then `picam.stop()` \u2192 apply `<NAME>`'s `gpio_state` \u2192 `os.system(i2c_cmd)` \u2192 `picam.start()` \u2192 `picam.capture_array("lores")`, bare/blocking. `setup/run_camera_prewarm.sh` invokes this once per catalog channel (A, B, C, D), each in its own short-lived process with its own external timeout.
    - **Superseded design (field-tested same day, abandoned):** an earlier `sweep` mode walked all of `['A', 'A', 'B', 'C', 'D']` inside **one** continuous process (the repeated 'A' mirrored legacy's `primer_ciclo` flag in `ventana_deteccion_movimiento.py`'s `PreviewThread.run()`, which forces an extra `stop()`/`start()` cycle on the very first loop iteration). Field testing (real hardware, `bash setup/start_iris.sh`) showed this **fails badly when a wedge hits mid-sweep**: channel A warmed up twice successfully, then the switch to B wedged, and since the whole process is a single blocking sequence with no in-process timeout, it hung until the wrapper's 60s group-kill fired \u2014 **channels C and D were never even attempted** that run, because one channel's wedge silently consumed the entire stage's time budget. Once each channel is its own process with its own timeout, this can't happen \u2014 every channel always gets its own independent attempt regardless of what happened to the others. The "repeat A" duplication was also dropped: it existed specifically to mimic `primer_ciclo` forcing an extra stop/start *within one continuous process*, which has no equivalent once every channel invocation is already a fresh process on its own.
  - **`setup/run_camera_prewarm.sh`** (shared by both boot paths): `run_prewarm_stage(timeout_s, ...)` launches `setsid python scripts/cold_boot_camera_prewarm.py "$@" &` (a new session so the PID doubles as the process group ID), polls `kill -0 "$pid"` once per second, and if the process is still alive after `timeout_s`, runs `kill -9 -- "-$pid"` \u2014 **the whole process *group*, not just the direct child** \u2014 since libcamera spawns its own IPA proxy as a separate PID that a plain single-PID kill would leave orphaned, still holding the camera device. Calls `run_prewarm_stage "$PROBE_TIMEOUT_S" probe`, then loops `run_prewarm_stage "$CHANNEL_TIMEOUT_S" channel "$channel"` over a fixed `A B C D` bash list (not read from `config/default_values.json` \u2014 must be kept in sync by hand if the MUX hardware ever changes). `PROBE_TIMEOUT_S=10`, `CHANNEL_TIMEOUT_S=8` (see "Plan v7" below). Reads `PREWARM_PYTHON` (env var, defaults to `python`) for the interpreter path, so `iris.service` (no venv activation) can override it while `start_iris.sh` (venv already activated) can rely on the default.
  - **Plans v5/v6 \u2014 timeout reduction attempts, both reverted (Sept 9 2026):** v5 cut a single `PREWARM_TIMEOUT_S` from 10s/8s down to 3s for both probe and channel, based on a consistent **1.3–1.5s** wedge-detection delta measured from libcamera's own internal log timestamps across 7+ real-hardware samples (essentially libcamera's hardcoded 1s V4L2 dequeue timeout plus ~0.3–0.4s of setup overhead). A follow-up field run showed `probe` hadn't wedged (or finished) by 3s while channel B/C/D still wedged at a consistent ~1.32s, so v6 split the timeout in two — `PROBE_TIMEOUT_S=8`, `CHANNEL_TIMEOUT_S=5` — and also tightened the polling loop from 1s to 0.2s granularity. **Both revisions were followed by a real-hardware failure** where Iris itself still wedged internally after prewarm finished (gunicorn worker crash-restart, same as before any of this tuning), and the v6 field run additionally reported **mismatched/mixed images between channels** in the final result \u2014 a new, more concerning symptom not seen with the original slower timeouts. This is strong evidence the wedge/restart problem is not caused by — and is not fixed by — tuning the prewarm script's timeouts at all; the real fix is still `CsiCameraAdapter.initialize_camera()`'s retry logic (see below), left deliberately untouched all this time.
  - **Plan v7 \u2014 revert to the original, known-safe budget (Sept 9 2026):** `PROBE_TIMEOUT_S=10`, `CHANNEL_TIMEOUT_S=8`, polling reverted from 0.2s back to 1s \u2014 exactly the values field-validated at 4/4 successful runs before the v5/v6 tuning experiments. Slower (worst case ~42s) but this is the configuration known to reach Iris reliably; keep it here until the underlying `CsiCameraAdapter` issue is addressed, at which point shorter prewarm timeouts can be revisited safely.
  - **\u26a0\ufe0f FROZEN as of `app/VERSION` 1.2.2 (Sept 9 2026) \u2014 do not re-tune these timeouts without new field evidence.** After repeated real-hardware regressions from shortening `PROBE_TIMEOUT_S`/`CHANNEL_TIMEOUT_S` (Plans v5/v6 above), the operator confirmed Plan v7's values (`PROBE_TIMEOUT_S=10`, `CHANNEL_TIMEOUT_S=8`, 1s polling) work reliably and asked to stop touching them ("as\u00ed funciona bien... ya no le muevas"). This exact configuration is what shipped in 1.2.2. If a future change to this script is needed, re-validate on real hardware before assuming shorter timeouts are safe again.
  - **Wired into both boot paths**, replacing the old bash retry loop entirely: `setup/start_iris.sh` runs `bash "$SCRIPT_DIR/run_camera_prewarm.sh"` after activating the venv; `setup/iris.service`'s `ExecStartPre=-/bin/bash -c 'PREWARM_PYTHON=__IRIS_HOME__/venvs/mezt/bin/python bash setup/run_camera_prewarm.sh'` (the leading `-` still tells systemd to ignore the overall exit code \u2014 the prewarm remains best-effort and must never block Iris from starting).
  - **Iris starts regardless of any stage/channel's outcome** \u2014 no retry of the whole prewarm sequence; this script/orchestrator's exit code is informational only, read from the logs, not acted upon by the caller.
  - **Field-tested finding that motivated this revision (Sept 8-9 2026, real hardware, `bash setup/start_iris.sh`):** with the earlier whole-sequence `sweep` design, after it gave up on B (and thus never touched C/D), Iris's own `CsiCameraAdapter.initialize_camera()` (3 attempts, fresh `Picamera2()` each attempt, already existed \u2014 unchanged by this fix) failed all 3 attempts on channel C with `Camera in Running state trying acquire() requiring state Available`, the unhandled exception killed the gunicorn worker (`workers = 1` in `gunicorn.conf.py`), gunicorn auto-respawned a fresh worker, and **that new process got one channel further (failed on D instead) before the log was cut off** \u2014 the same "each fresh process clears one more channel" pattern from the original 2026-09-09 finding (see above), just relocated from the boot-time prewarm script into Iris's own gunicorn crash-restart cycle, and apparently not reliably converging within a reasonable number of automatic restarts (the operator still had to manually restart Iris several times, same complaint as before this whole redesign saga started).
  - **Still not fully understood / explicitly deferred by the operator's own choice this round:** `CsiCameraAdapter.initialize_camera()`'s retry logic (`hw_lock`, `sleep(0.5)`/`sleep(1)` between attempts) was deliberately left unchanged this round, even though it is simpler and shorter (5 attempts, `sleep(0.02)`, no lock) in legacy's `camara.py Inicializar()` and legacy does not appear to hit the same "Running state" acquire failure in the field \u2014 flagged as a likely next thing to revisit if per-channel prewarm alone does not fully resolve the manual-restart complaint.
  - **Validated on real hardware (Sept 9 2026):** (a) confirmed \u2014 every one of A/B/C/D always gets attempted regardless of earlier channels' outcomes; (b) confirmed \u2014 `log_camera_diagnostics()` (automatic `lsof /dev/video0` + `pgrep -af rpi/vc4` after every kill and once at the end) shows the device and IPA proxy are always cleanly released, across 8+ independent kill events in 2 separate log submissions; (c) confirmed \u2014 4/4 consecutive full runs (`bash setup/start_iris.sh`) reached Iris without any manual restart, even with 3/4 channels failing prewarm on some runs, and without triggering the gunicorn worker crash-restart loop described above; (d) confirmed \u2014 skipping straight to channel A alone (no B/C/D) does reproduce the manual-restart failure, so all 4 channels are genuinely required, not excessive.
  - **Iris's own startup no longer re-warms the channels right after this script runs:** `IrisServer._load_sequence(state, path, warmup=False)` (default) skips `AppFactory.warmup_all_channels()` on the two fully-automatic reload paths \u2014 the initial sequence pre-load at process startup and `_resume_session_if_any()`'s auto-resume after a restart \u2014 specifically so the per-channel state left by this prewarm script stays observable in the UI/logs instead of being immediately overwritten. Explicit operator actions (Load Sequence, Edit sequence, Clone as new part, Resume draft) still pass `warmup=True` to refresh the preview immediately, since there the operator is actively watching for it.
  - **Deliberately out of scope for now:** `CsiCameraAdapter.py`/`RpiCsiMuxAdapter.py`/`AppFactory.py`/`IrisServer.py` are untouched by this fix \u2014 the goal is to resolve the hardware state *before* Iris/gunicorn ever touch it, without yet changing Iris's own adapter architecture. If the wedge still reproduces *inside* Iris itself after this prewarm reliably succeeds on its own, the next suspect is `CsiCameraAdapter`'s `hw_lock`/`_call_picam_with_timeout` layer.
- **Reverted in-process attempt (superseded by the fix above, kept here for context in case it resurfaces):** `AppFactory._get_system_uptime_s()` / `_is_cold_boot(threshold_s=120.0)` (read `/proc/uptime`) and `_has_csi_mux` gated a `_warmup_all_channels_cold_boot()` branch inside `warmup_all_channels()` that looped over `self._defaults["csi_channels"]` calling `mux.select_channel_gpio_only()` \u2192 `camera.initialize_camera()` \u2192 `capture_frame()` **all inside the same Iris process**. Confirmed on real hardware that this does not clear the wedge (see "The Issue" above) \u2014 removed entirely; `warmup_all_channels()` is back to the plain fast sweep (`capture_preview_frame(channel, settle_delay=0.1)` per `get_camera_ports()` channel) it had before this investigation.
- **Cleanup of prior experiments that did not help (confirmed on real hardware, removed as part of this fix):**
  - **`use_lores` parameter** (an earlier experiment that tried reading the lores/YUV420 stream instead of main during warm-up, hypothesizing it would avoid ISP timeouts): tested on real hardware, produced the **same** wedge pattern \u2014 removed entirely from `ICamera.capture_frame()`, `CsiCameraAdapter.capture_frame()`, `UsbCameraAdapter.capture_frame()`, and `AppFactory.capture_preview_frame()`/`warmup_all_channels()`. `capture_frame()` always reads the `main` stream again.
  - **`settle_delay=0.3` in the fast-sweep `warmup_all_channels()`** (tripled from the original `0.1` s in an earlier attempt): also confirmed to have no effect on the wedge \u2014 reverted back to `0.1` s.
  - **Diagnostic logging (`_log_power_status()`, `_log_hardware_diagnostics()`)**: both static methods remain defined in `AppFactory` (useful for future debugging), but their call sites in `initialize_hardware()` and `_recover_camera_after_failure()` are now commented out (dormant), not deleted \u2014 they added console noise without changing behavior once the cold-boot root cause was identified as kernel-level, not something these commands could diagnose further in production.
- **What was intentionally left unchanged:** `capture_frame()`'s single-attempt-then-abandon design (no internal retry loop) \u2014 still required because `_call_picam_with_timeout()`'s fire-and-forget daemon thread cannot be cancelled, so retrying against the same instance would race a second concurrent native call. The `hw_lock`/`RLock` fix and `_PICAM_CALL_TIMEOUT_S = 1.0` (section 23) are unrelated to this cold-boot issue and were not touched.

### 25. Per-View `inference_type`: Presence Detection with Absence-Calibrated Baseline (Sept 2026, shipped in `app/VERSION` 1.2.2)
- **The Feature:** A new optional `"inference_type"` field on each entry of `preprocessing_image_parameters` in the sequence JSON. Two values are supported:
  - `"standard"` (default, used when the field is absent \u2014 fully backward compatible with every sequence JSON written before this feature) \u2014 the existing behavior: `is_ok = threshold_min <= score <= threshold_max`.
  - `"presence_detection_absence_calibrated"` \u2014 for views where PaDiM is calibrated on the **absent** state of a feature (e.g. a Water Pump M02 oil-drop-presence check, calibrated on "no oil" images) and an anomaly (score outside the calibrated range) means the feature is **present**, which is the desired/OK outcome. The final decision is simply inverted: `is_ok = not (threshold_min <= score <= threshold_max)`.
- **The single inversion point:** `InspectionService.execute_full_inspection_from_frames()` \u2014 right after computing `is_ok` from the score/threshold comparison, it flips it when `self._settings.get_inference_type(view_name) == "presence_detection_absence_calibrated"`. Every downstream consumer (`Part.overall_status`, NOK GPIO dispatch steps, `LocalStorageAdapter` traceability JSONL, `TraceabilityReviewService`, the heatmap endpoint) reads only the already-inverted `is_ok`/`classification`, so **no other runtime code needed changes** \u2014 confirmed by direct code reading, not assumed:
  - `TraceabilityReviewService._find_nok_images()` filters/sorts on the already-inverted `classification` string, so its descending-score hard-example-mining sort remains semantically correct for inverted views too (it still selects the most borderline "no oil" images to reinforce `train/OK`).
  - `GET /api/heatmap/<view_name>` normalizes each error map independently (`cv2.normalize(..., NORM_MINMAX)`), agnostic to `is_ok`/thresholds \u2014 needs no changes.
  - `CalibrationService`'s sweep/fit algorithm is unchanged \u2014 `train_ok`/`test_ok` folders are still "absent-feature" (no-oil) baseline images and `test_nok` is still "present-feature" (with-oil) validation-only images, **exactly like every other, non-inverted view**. This is only metadata propagation for audit purposes, not a different calibration algorithm.
- **\u26a0\ufe0f Calibration folder convention is completely independent of the runtime inversion \u2014 never invert it too.** `train/OK` and `test/OK` must always contain the **absent**-feature (no-oil) images, and `test/NOK` must always contain the **present**-feature (with-oil) validation-only images, for both `"standard"` and `"presence_detection_absence_calibrated"` views alike. Putting "with oil" images in `test/OK` would corrupt `threshold_max`/the Gaussian fit by widening the "normal" range to include the anomalous pattern, defeating the whole purpose of the calibration.
- **Metadata propagation (audit trail, does not affect the fit):**
  - `SequenceSettings`: `_inference_types`/`_port_inference_types` dicts (populated in `__init__` alongside `_pipelines`/`_port_pipelines`) and `get_inference_type(view_name)` \u2014 same 3-level fallback lookup (exact `view_name`, then prefix without channel suffix, then bare `camera_port`) as `preprocess_for_inference()`.
  - `ViewConfigBuilder.build_view_configs()` includes `"inference_type"` per view config (default `"standard"`).
  - `CalibrationModels.CalibrationResult` gained `inference_type: str = "standard"`; `CalibrationService.run_calibration()` populates it from the view config and merges a new `"inference_types": {view_name: inference_type}` dict into `eval_config.json`, following the exact same merge-with-existing pattern already used for `blocks`/`backbone_paths` (so a partial recalibration never discards untouched views' `inference_type`). `IrisServer._write_calibration_eval()` includes the same field in `calibration_eval.json`.
  - `SequenceBuilderService.update_pipeline()` gained an `inference_type: str = "standard"` parameter, set on both the "replace existing entry" and "append new entry" branches. `POST /api/builder/update_pipeline` reads `inference_type` from the request body (defaults to `"standard"` if omitted, so old frontend clients/API callers keep working unchanged).
  - `builder.html`/`builder.js`: a new "Inference type" `<select>` in the pipeline panel (`#inference-type-select`, options "Standard" / "Presence detection (absence-calibrated, invert OK/NOK)"), backed by a per-section/per-channel `inferenceTypes` state object mirroring the existing `pipelines` object \u2014 restored from the draft's `preprocessing_image_parameters[].inference_type` on load, synced to the select whenever the active camera/section changes (`syncInferenceTypeSelect()`), and sent as part of every `/api/builder/update_pipeline` payload (`savePipeline()`).

