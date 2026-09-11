/**
 * builder.js — Iris sequence builder logic.
 *
 * Depends on: canvas_tools.js (CanvasTools), DRAFT global (injected by template).
 *
 * Responsibilities:
 *   - Create one CanvasTools instance per camera channel.
 *   - Manage sections (multiple preprocessing views of the same part).
 *   - Send pipeline updates, step add/remove to the server.
 *   - Drive the step editor modal.
 *   - Trigger draft auto-save on every mutation.
 */

(function () {
    "use strict";

    // ── State ─────────────────────────────────────────────────────────────
    const draft = DRAFT;               // injected by template
    const cameraPorts = draft.hardware?.camera_port || [];
    let captureW = (draft.hardware?.camera_capture_resolution || [4608, 2592])[0];
    let captureH = (draft.hardware?.camera_capture_resolution || [4608, 2592])[1];

    let sections = [];    // [{id, label}]
    let activeSectionId = null;
    let activeChannel = null;  // camera port currently selected
    const canvasMap = {};    // channel -> CanvasTools
    // pipelines[sectionId][channel] = [{tool, parameters?}, ...]
    const pipelines = {};
    // inferenceTypes[sectionId][channel] = "standard" | "presence_detection_absence_calibrated"
    const inferenceTypes = {};

    // ── Init ──────────────────────────────────────────────────────────────

    document.addEventListener("DOMContentLoaded", function () {
        buildCameraGrid();
        bindHardwareSettings();
        bindToolButtons();
        bindPipelineButtons();
        bindStepButtons();
        bindSaveValidate();
        showInferenceReminder();

        // ResizeObserver keeps every canvas matched to its card whenever the
        // layout changes — window resize, browser zoom, panel collapse, etc.
        // Falls back gracefully on browsers without ResizeObserver support.
        if (typeof ResizeObserver !== "undefined") {
            var ro = new ResizeObserver(function () { resizeAllCanvases(); });
            cameraPorts.forEach(function (ch) {
                var card = document.getElementById("card-" + ch);
                if (card) ro.observe(card);
            });
        }

        // Defer canvas sizing and draft restoration until after the first
        // browser layout pass so card dimensions are known. resizeAllCanvases()
        // must run before restoreSectionsFromDraft() so that importPipeline()
        // uses the correct display dimensions for coordinate scaling.
        requestAnimationFrame(function () {
            resizeAllCanvases();
            restoreSectionsFromDraft();
            setupSpotlights();
            if (sections.length === 0) addSection();
            if (cameraPorts.length > 0) selectChannel(cameraPorts[0]);
        });
    });

    // ── Camera grid ───────────────────────────────────────────────────────

    function buildCameraGrid() {
        const grid = document.getElementById("camera-grid");
        grid.className = "camera-grid cameras-" + cameraPorts.length;

        cameraPorts.forEach(function (ch) {
            // Card
            const card = document.createElement("div");
            card.className = "camera-card";
            card.id = "card-" + ch;

            const lbl = document.createElement("span");
            lbl.className = "camera-card-label";
            lbl.textContent = "Camera " + ch;
            card.appendChild(lbl);

            // Canvas element
            const canvas = document.createElement("canvas");
            canvas.id = "canvas-" + ch;
            // Set canvas pixel dimensions to match card display size
            canvas.width = 480;
            canvas.height = 270;
            card.appendChild(canvas);

            card.addEventListener("click", function () {
                selectChannel(ch);
            });

            grid.appendChild(card);

            // CanvasTools instance (initial 480×270; resized to actual card
            // dimensions in resizeAllCanvases() before DOMContentLoaded returns).
            canvasMap[ch] = new CanvasTools("canvas-" + ch, captureW, captureH);
            canvasMap[ch].onShapeAdded = function () { renderPipelineList(); };
        });
    }

    function selectChannel(ch) {
        activeChannel = ch;
        document.querySelectorAll(".camera-card").forEach(function (c) {
            c.classList.toggle("selected", c.id === "card-" + ch);
        });
        document.getElementById("active-camera-label").textContent = "Camera " + ch;
        document.getElementById("pipeline-camera-label").textContent = "Cam " + ch;
        renderPipelineList();
        syncInferenceTypeSelect();
    }

    // ── Sections ──────────────────────────────────────────────────────────

    function restoreSectionsFromDraft() {
        const params = draft.preprocessing_image_parameters || [];
        const sectionIds = [...new Set(params.map(function (e) { return e.section; }))];
        sectionIds.forEach(function (sid) {
            const id = parseInt(sid, 10) || sections.length + 1;
            addSection(id, false);   // false = don't auto-save
            cameraPorts.forEach(function (ch) {
                const entry = params.find(function (e) {
                    return e.camera_port === ch && e.section === sid;
                });
                if (entry) {
                    pipelines[id] = pipelines[id] || {};
                    pipelines[id][ch] = entry.pipeline || [];
                    canvasMap[ch].importPipeline(entry.pipeline || []);
                    inferenceTypes[id] = inferenceTypes[id] || {};
                    inferenceTypes[id][ch] = entry.inference_type || "standard";
                }
            });
        });
    }

    function addSection(id, autoSave) {
        const sectionId = id || (sections.length + 1);
        sections.push({ id: sectionId, label: "Section " + sectionId });
        pipelines[sectionId] = pipelines[sectionId] || {};
        inferenceTypes[sectionId] = inferenceTypes[sectionId] || {};
        cameraPorts.forEach(function (ch) {
            pipelines[sectionId][ch] = pipelines[sectionId][ch] || [];
            inferenceTypes[sectionId][ch] = inferenceTypes[sectionId][ch] || "standard";
        });
        renderSectionTabs();
        if (activeSectionId === null) setActiveSection(sectionId);
        if (autoSave !== false) saveDraftSilently();
    }

    function setActiveSection(id) {
        activeSectionId = id;
        document.getElementById("capture-section-label").textContent = "Section " + id;
        renderSectionTabs();
        // Reload each canvas from the stored pipeline for this section
        cameraPorts.forEach(function (ch) {
            canvasMap[ch].clearObjects();
            const pipe = (pipelines[id] || {})[ch] || [];
            canvasMap[ch].importPipeline(pipe);
        });
        renderPipelineList();
        syncInferenceTypeSelect();
    }

    function syncInferenceTypeSelect() {
        const select = document.getElementById("inference-type-select");
        if (!select || !activeChannel || activeSectionId === null) return;
        select.value = (inferenceTypes[activeSectionId] || {})[activeChannel] || "standard";
    }

    function renderSectionTabs() {
        const container = document.getElementById("section-tabs");
        container.innerHTML = "";
        sections.forEach(function (sec) {
            const el = document.createElement("div");
            el.className = "tab-item" + (sec.id === activeSectionId ? " active" : "");
            el.textContent = sec.label;
            el.addEventListener("click", function () { setActiveSection(sec.id); });
            container.appendChild(el);
        });
    }

    document.getElementById("btn-add-section").addEventListener("click", function () {
        addSection();
    });

    // ── Tool palette ──────────────────────────────────────────────────────

    function bindToolButtons() {
        document.querySelectorAll(".tool-btn").forEach(function (btn) {
            btn.addEventListener("click", function () {
                const tool = btn.dataset.tool;
                document.querySelectorAll(".tool-btn").forEach(function (b) {
                    b.classList.remove("active");
                });
                btn.classList.add("active");
                if (activeChannel) canvasMap[activeChannel].setTool(tool);
            });
        });
    }

    // ── Capture preview ───────────────────────────────────────────────────

    // ── Canvas resize ─────────────────────────────────────────────────────

    function resizeAllCanvases() {
        cameraPorts.forEach(function (ch) {
            const card = document.getElementById("card-" + ch);
            if (!card || !canvasMap[ch]) return;
            const w = card.clientWidth;
            const h = card.clientHeight;
            if (w > 10 && h > 10) canvasMap[ch].resize(w, h);
        });
    }

    // ── Capture preview ───────────────────────────────────────────────────

    document.getElementById("btn-capture").addEventListener("click", function () {
        const status = document.getElementById("capture-status");
        status.textContent = "Capturing…";

        fetch("/api/builder/capture_preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ section: activeSectionId }),
        })
            .then(function (r) { return r.json(); })
            .then(function (d) {
                const captured = d.captured || [];
                const errors = d.errors || {};
                captured.forEach(function (ch) {
                    const card = document.getElementById("card-" + ch);
                    if (card) card.classList.toggle("disconnected", !!errors[ch]);
                });
                if (d.hardware_wedged_message) showCriticalBanner(d.hardware_wedged_message);
                const loadPromises = captured
                    .filter(function (ch) { return !errors[ch]; })
                    .map(function (ch) {
                        return fetch("/api/builder/frame/" + ch)
                            .then(function (r) { return r.blob(); })
                            .then(function (blob) {
                                if (canvasMap[ch]) canvasMap[ch].setBackground(blob);
                            });
                    });
                return Promise.all(loadPromises).then(function () {
                    const errKeys = Object.keys(errors);
                    status.textContent = errKeys.length > 0
                        ? "Captured with errors: " + errKeys.join(", ")
                        : "Preview updated.";
                });
            })
            .catch(function () { status.textContent = "Capture failed."; });
    });

    // ── Critical alarm banner (latched — only cleared by a service restart) ─

    function showCriticalBanner(msg) {
        if (document.getElementById("critical-error-banner")) return;  // already shown
        const el       = document.createElement("div");
        el.id          = "critical-error-banner";
        el.className   = "alert alert-error inpage";
        el.textContent = "⚠ " + msg;
        document.querySelector(".capture-toolbar").after(el);
    }

    // ── Reference image (offline ROI setup without live camera) ──────────
    // Loads a local image file as the canvas background for the active camera.
    // The background persists when switching sections, so one load per camera
    // is enough even with multiple sections.

    (function () {
        const btnRef   = document.getElementById("btn-load-ref");
        const fileInput = document.getElementById("ref-img-input");
        const status   = document.getElementById("capture-status");
        if (!btnRef || !fileInput) return;

        btnRef.addEventListener("click", function () {
            if (!activeChannel) {
                status.textContent = "Select a camera card first.";
                return;
            }
            // Reset so the same file can be re-picked after a clear
            fileInput.value = "";
            fileInput.click();
        });

        fileInput.addEventListener("change", function () {
            const file = fileInput.files[0];
            if (!file || !activeChannel) return;
            canvasMap[activeChannel].setBackground(file);
            status.textContent = "Reference image loaded for camera " + activeChannel + ".";
        });
    }());

    // ── Pipeline panel ────────────────────────────────────────────────────

    function bindPipelineButtons() {
        document.getElementById("btn-add-inference-type").addEventListener("click", function () {
            if (!activeChannel || activeSectionId === null) return;
            const value = document.getElementById("inference-type-select").value;
            inferenceTypes[activeSectionId] = inferenceTypes[activeSectionId] || {};
            inferenceTypes[activeSectionId][activeChannel] = value;
            savePipeline(activeChannel, activeSectionId, getPipeline(activeChannel, activeSectionId));
        });

        document.getElementById("btn-apply-pipeline").addEventListener("click", function () {
            if (!activeChannel || activeSectionId === null) return;
            const ct = canvasMap[activeChannel];
            const canvasPipe = ct.exportPipeline();

            // Find if exist a ROI in canvas.
            let targetW = 525;
            let targetH = 525;
            let roiFound = false;

            for (let i = 0; i < canvasPipe.length; i++) {
                if (canvasPipe[i].tool === "apply_roi_crop") {
                    targetW = canvasPipe[i].parameters.w;
                    targetH = canvasPipe[i].parameters.h;
                    roiFound = true;
                    break;
                }
            }

            // If no ROI found, use the values from the resize inputs or defaults (525×525)
            if (!roiFound) {
                targetW = parseInt(document.getElementById("resize-w").value, 10) || 525;
                targetH = parseInt(document.getElementById("resize-h").value, 10) || 525;
            } else {
                // If ROI found, update the resize inputs to match the ROI dimensions
                document.getElementById("resize-w").value = targetW;
                document.getElementById("resize-h").value = targetH;
            }

            // Add resize to the end of the pipeline.
            canvasPipe.push({
                tool: "resize_to_training_resolution",
                parameters: { width: targetW, height: targetH },
            });

            // Rescue camera settings to not erase them.
            const oldPipe = getPipeline(activeChannel, activeSectionId);
            const camSettings = oldPipe.filter(function (t) {
                return t.tool == "set_time_exposure" || t.tool === "set_lens_position";
            });

            // Mix camera settings with the rest of the pipeline and save.
            const newPipe = camSettings.concat(canvasPipe);
            savePipeline(activeChannel, activeSectionId, newPipe);
        });

        document.getElementById("btn-add-cam-settings").addEventListener("click", function () {
            if (!activeChannel || activeSectionId === null) return;
            const exp = parseInt(document.getElementById("cam-exposure").value, 10);
            const lens = parseFloat(document.getElementById("cam-lens").value);
            const pipe = getPipeline(activeChannel, activeSectionId);

            const filtered = pipe.filter(function (t) {
                return t.tool !== "set_time_exposure" && t.tool !== "set_lens_position";
            });

            if (exp > 0) filtered.unshift({ tool: "set_time_exposure", parameters: { exposure_time: exp } });
            if (lens > 0) filtered.unshift({ tool: "set_lens_position", parameters: { lens_position: lens } });
            savePipeline(activeChannel, activeSectionId, filtered);
        });

        document.getElementById("btn-add-resize").addEventListener("click", function () {
            if (!activeChannel || activeSectionId === null) return;
            const rw = parseInt(document.getElementById("resize-w").value, 10);
            const rh = parseInt(document.getElementById("resize-h").value, 10);
            const pipe = getPipeline(activeChannel, activeSectionId);
            // Remove existing resize
            const filtered = pipe.filter(function (t) { return t.tool !== "resize_to_training_resolution"; });
            filtered.push({ tool: "resize_to_training_resolution", parameters: { width: rw, height: rh } });
            savePipeline(activeChannel, activeSectionId, filtered);
        });
    }

    function getPipeline(ch, sectionId) {
        return JSON.parse(JSON.stringify((pipelines[sectionId] || {})[ch] || []));
    }

    function savePipeline(ch, sectionId, pipe) {
        const sec = sections.find(function (s) { return s.id === sectionId; });
        if (!sec) return;

        pipelines[sectionId][ch] = pipe;
        renderPipelineList();

        fetch("/api/builder/update_pipeline", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                camera_port: ch,
                view: "section_" + sectionId,   // server expects "section_{id}"
                section: String(sectionId),
                pipeline: pipe,
                inference_type: (inferenceTypes[sectionId] || {})[ch] || "standard",
            }),
        }).catch(function () { console.error("Failed to save pipeline."); });
    }

    function renderPipelineList() {
        const list = document.getElementById("pipeline-list");
        list.innerHTML = "";
        if (!activeChannel || activeSectionId === null) return;
        const pipe = getPipeline(activeChannel, activeSectionId);
        updateResizeHighlight(pipe);
        if (pipe.length === 0) {
            list.innerHTML = '<li class="pipeline-empty muted">No tools yet.</li>';
            return;
        }
        pipe.forEach(function (tool, idx) {
            const li = document.createElement("li");
            li.className = "pipeline-item";
            // Make parameters responsive - each on new line if needed
            let paramsHTML = "";
            if (tool.parameters) {
                const entries = Object.entries(tool.parameters);
                paramsHTML = entries.map(function (kv) {
                    return `<span class="param-pair">${kv[0]}: ${kv[1]}</span>`;
                }).join(" ");
            }
            li.innerHTML = `
                <span class="step-num">${idx + 1}</span>
                <span class="tool-name">${tool.tool}</span>
                <span class="tool-args">${paramsHTML}</span>
                <span class="remove-tool" data-idx="${idx}">✕</span>
            `;
            li.querySelector(".remove-tool").addEventListener("click", function () {
                const newPipe = getPipeline(activeChannel, activeSectionId);
                newPipe.splice(idx, 1);
                savePipeline(activeChannel, activeSectionId, newPipe);
            });
            list.appendChild(li);
        });
    }

    // Missing resize is the most common "forgot to configure" mistake — keep it visually loud until fixed.
    function updateResizeHighlight(pipe) {
        const section = document.getElementById("section-resize");
        if (!section) return;
        const hasResize = pipe.some(function (t) { return t.tool === "resize_to_training_resolution"; });
        section.classList.toggle("panel-section-warn", !hasResize);
    }

    // ── Steps ─────────────────────────────────────────────────────────────

    function bindStepButtons() {
        document.getElementById("btn-add-step").addEventListener("click", openStepModal);
        document.getElementById("btn-save-step").addEventListener("click", saveStep);
    }

    function openStepModal(stepData) {
        const modal = document.getElementById("step-modal");
        modal.classList.remove("hidden");

        // Clear editing mode flag when opening fresh
        const btnSave = document.getElementById("btn-save-step");
        delete btnSave.dataset.editingStep;

        if (stepData && typeof stepData === "object") {
            document.getElementById("step-num").value = stepData.step_number || 1;
            document.getElementById("step-desc").value = stepData.description || "";
        } else {
            document.getElementById("step-num").value = nextStepNumber();
            document.getElementById("step-desc").value = "";
        }
        renderStepFields();
    }

    function closeStepModal() {
        document.getElementById("step-modal").classList.add("hidden");
    }
    // expose for template onclick
    window.closeStepModal = closeStepModal;

    function nextStepNumber() {
        const steps = draft.steps || [];
        const nums = steps.map(function (s) { return s.step_number; })
            .filter(function (n) { return n >= 0; });
        return nums.length > 0 ? Math.max.apply(null, nums) + 1 : 1;
    }

    // expose for template onchange
    window.renderStepFields = function () {
        const type = document.getElementById("step-type").value;
        const fields = document.getElementById("step-type-fields");
        const stepN = parseInt(document.getElementById("step-num").value, 10);

        if (type === "gpio" || type === "nok") {
            // Auto-set step number sign
            if (type === "nok" && stepN >= 0) {
                document.getElementById("step-num").value = -Math.abs(stepN || 1);
            }
            const pinsOptions = buildGpioPinsOptions(type === "nok" ? "output" : null);
            fields.innerHTML = `
                <div class="field-group">
                    <label>GPIO pin (BCM)</label>
                    <select id="step-gpio-pin">${pinsOptions}</select>
                </div>
                <div class="field-group">
                    <label>Action</label>
                    <select id="step-gpio-action">
                        <option value="turn_on">turn_on</option>
                        <option value="turn_off">turn_off</option>
                        <option value="send_output">send_output</option>
                        <option value="wait_for_input">wait_for_input</option>
                    </select>
                </div>
                
                <div class="field-row hidden" id="step-gpio-wait-params">
                    <div class="field-group">
                        <label>Expected Signal</label>
                        <select id="step-gpio-expected">
                            <option value="1">1 (High / Voltage)</option>
                            <option value="0">0 (Low / 0V)</option>
                        </select>
                    </div>
                    <div class="field-group">
                        <label>Timeout (ms)</label>
                        <input type="number" id="step-gpio-timeout" value="8000" min="0">
                        <small class="muted" style="display:block; margin-top:4px;">0 = Infinite</small>
                    </div>
                </div>

                <div class="field-group" id="step-delay-group">
                    <label>Delay after step (ms)</label>
                    <input type="number" id="step-delay" value="0" min="0">
                </div>
            `;
            
            // Show extra fields only if action is wait_for_input
            const actionSelect = document.getElementById("step-gpio-action");
            actionSelect.addEventListener("change", function() {
                const waitParams = document.getElementById("step-gpio-wait-params");
                if (waitParams) {
                    if (this.value === "wait_for_input") {
                        waitParams.classList.remove("hidden");
                    } else {
                        waitParams.classList.add("hidden");
                    }
                }
            });
            // Execute once to initialize the view
            actionSelect.dispatchEvent(new Event("change"));

        } else if (type === "camera") {
            const portOptions = cameraPorts.map(function (p) {
                return `<option value="${p}">Camera ${p}</option>`;
            }).join("");
            const sectionOptions = sections.map(function (s) {
                return `<option value="section_${s.id}">${s.label}</option>`;
            }).join("");
            fields.innerHTML = `
                <div class="field-group">
                    <label>Camera port</label>
                    <select id="step-cam-port">${portOptions}</select>
                </div>
                <div class="field-group">
                    <label>Belongs to Section</label>
                    <select id="step-view-prefix">${sectionOptions}</select>
                    <small class="muted">Link this capture to a section.</small>
                </div>
                <div class="field-group" id="step-delay-group">
                    <label>Delay after step (ms)</label>
                    <input type="number" id="step-delay" value="0" min="0">
                </div>
            `;
        } else if (type === "detect_piece") {
            const portOptions = cameraPorts.map(function (p) {
                return `<option value="${p}">Camera ${p}</option>`;
            }).join("");
            fields.innerHTML = `
                <div class="field-group">
                    <label>Camera port</label>
                    <select id="step-detect-port">${portOptions}</select>
                </div>
                <div class="field-group">
                    <label>Darkness threshold (0–255)</label>
                    <input type="number" id="step-detect-threshold" value="80" min="0" max="255">
                    <small class="muted">Mean brightness below this → piece detected. Draw the Detection ROI (orange) on the canvas first.</small>
                </div>
                <div class="field-group" id="step-delay-group">
                    <label>Delay after step (ms)</label>
                    <input type="number" id="step-delay" value="0" min="0">
                </div>
            `;
        } else if (type === "wait_for_piece") {
            const portOptions = cameraPorts.map(function (p) {
                return `<option value="${p}">Camera ${p}</option>`;
            }).join("");
            fields.innerHTML = `
                <div class="field-group">
                    <label>Camera port</label>
                    <select id="step-wfp-port">${portOptions}</select>
                </div>
                <div class="field-group">
                    <label>Pixel diff threshold (0–255)</label>
                    <input type="number" id="step-wfp-threshold" value="15" min="1" max="255">
                    <small class="muted">
                        Minimum mean pixel change (grayscale) needed to detect the piece.<br>
                        <strong>Higher</strong> → less sensitive, only large changes trigger (fewer false detections).<br>
                        <strong>Lower</strong> → more sensitive, small changes trigger (risk of false detections from lighting noise).<br>
                        Typical starting point: 15. Increase if it triggers without a piece; decrease if it misses the piece.
                    </small>
                </div>
                <div class="field-group">
                    <label>Poll interval (ms)</label>
                    <input type="number" id="step-wfp-poll" value="100" min="10" max="5000">
                    <small class="muted">
                        How often the camera is sampled while waiting. Common values:<br>
                        33 ms ≈ 30 fps &nbsp;|&nbsp; 67 ms ≈ 15 fps &nbsp;|&nbsp; 100 ms ≈ 10 fps &nbsp;|&nbsp; 200 ms ≈ 5 fps<br>
                        Recommended: 100–200 ms for industrial triggers (no need for video-rate polling).
                    </small>
                </div>
                <div class="field-group">
                    <label>Timeout (ms, 0 = indefinite)</label>
                    <input type="number" id="step-wfp-timeout" value="0" min="0">
                </div>
                <div class="field-group">
                    <label>Stabilization delay (ms)</label>
                    <input type="number" id="step-wfp-stabilization" value="0" min="0">
                    <small class="muted">Wait after MUX channel switch before capturing the reference frame.</small>
                </div>
                <div class="field-group" id="step-delay-group">
                    <label>Delay after step (ms)</label>
                    <input type="number" id="step-delay" value="0" min="0">
                </div>
            `;
        } else if (type === "inference") {
            document.getElementById("step-num").value = 1001;
            fields.innerHTML = `<p class="muted" style="font-size:.85rem">
                Inference step runs InspectionService on all captured frames.<br>
                step_number must be ≥ 1001.
            </p>
            <div class="field-group" id="step-delay-group">
                <label>Delay after step (ms)</label>
                <input type="number" id="step-delay" value="0" min="0">
            </div>`;
        }
    };

    function buildGpioPinsOptions(filterType) {
        const gpioConf = (draft.hardware || {}).gpio_configuration || [];
        return gpioConf
            .filter(function (g) { return !filterType || g.type === filterType; })
            .map(function (g) {
                return `<option value="${g.pin_number}">BCM ${g.pin_number} — ${g.description || g.type}</option>`;
            }).join("");
    }

    function saveStep() {
        const type = document.getElementById("step-type").value;
        const num = parseInt(document.getElementById("step-num").value, 10);
        const desc = document.getElementById("step-desc").value.trim();
        const step = { step_number: num, description: desc };

        if (type === "gpio" || type === "nok") {
            const pin = parseInt(document.getElementById("step-gpio-pin").value, 10);
            const action = document.getElementById("step-gpio-action").value;
            const delay = parseInt(document.getElementById("step-delay").value || "0", 10);
            step.gpio_action = [{ pin_number: pin, action: action }];
            if (delay > 0) step.delay_after_step = delay;

            if (action === "wait_for_input") {
                const timeoutInput = document.getElementById("step-gpio-timeout");
                const expectedInput = document.getElementById("step-gpio-expected");
                
                step.wait_for_input_parameters = {};
                
                if (timeoutInput && timeoutInput.value !== "") {
                    step.wait_for_input_parameters.timeout = parseInt(timeoutInput.value, 10);
                }
                if (expectedInput && expectedInput.value !== "") {
                    step.wait_for_input_parameters.expected_value = parseInt(expectedInput.value, 10);
                }
            }

        } else if (type === "camera") {
            const port = document.getElementById("step-cam-port").value;
            const prefix = document.getElementById("step-view-prefix").value.trim();
            const delay = parseInt(document.getElementById("step-delay").value || "0", 10);
            step.camera_action = [{ camera_port: port, prefix_view: prefix }];
            if (delay > 0) step.delay_after_step = delay;

        } else if (type === "detect_piece") {
            const port = document.getElementById("step-detect-port").value;
            const threshold = parseInt(document.getElementById("step-detect-threshold").value || "80", 10);
            const ct = canvasMap[port];
            const roi = ct ? ct.exportDetectionRoi() : null;
            if (!roi) {
                alert("Draw a Detection ROI (orange rectangle) on camera " + port + " before saving this step.");
                return;
            }
            const detDelay = parseInt(document.getElementById("step-delay").value || "0", 10);
            step.detect_piece_action = {
                camera_port: port,
                darkness_threshold: threshold,
                roi: roi,
            };
            if (detDelay > 0) step.delay_after_step = detDelay;

        } else if (type === "wait_for_piece") {
            const port = document.getElementById("step-wfp-port").value;
            const ct = canvasMap[port];
            const roi = ct ? ct.exportDetectionRoi() : null;
            if (!roi) {
                alert("Draw a Detection ROI (orange rectangle) on camera " + port + " before saving this step.");
                return;
            }
            const threshold = parseInt(document.getElementById("step-wfp-threshold").value || "15", 10);
            const poll = parseInt(document.getElementById("step-wfp-poll").value || "100", 10);
            const timeout = parseInt(document.getElementById("step-wfp-timeout").value || "0", 10);
            const stabilization = parseInt(document.getElementById("step-wfp-stabilization").value || "0", 10);
            const wfpDelay = parseInt(document.getElementById("step-delay").value || "0", 10);
            step.wait_for_piece_action = {
                camera_port: port,
                roi: roi,
                pixel_diff_threshold: threshold,
                poll_interval_ms: poll,
                timeout_ms: timeout,
                stabilization_ms: stabilization,
            };
            if (wfpDelay > 0) step.delay_after_step = wfpDelay;

        } else if (type === "inference") {
            step.step_number = Math.max(step.step_number, 1001);
            step.inference_action = ["execute_full_inspection"];
            const infDelay = parseInt(document.getElementById("step-delay").value || "0", 10);
            if (infDelay > 0) step.delay_after_step = infDelay;
        }

        // Check if editing an existing step
        const btnSave = document.getElementById("btn-save-step");
        const editingStepNum = btnSave.dataset.editingStep;

        // If editing and step number changed, remove old step first
        const savePromise = (editingStepNum && parseInt(editingStepNum) !== num)
            ? fetch("/api/builder/remove_step", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ step_number: parseInt(editingStepNum) }),
            }).then(function () { return true; })
            : Promise.resolve(true);

        savePromise.then(function () {
            return fetch("/api/builder/add_step", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ step }),
            });
        })
            .then(function (r) { return r.json(); })
            .then(function (d) {
                if (d.ok) {
                    draft.steps = d.steps;
                    renderStepsTree();
                    showInferenceReminder();
                    closeStepModal();
                    delete btnSave.dataset.editingStep; // Clear editing mode
                }
            })
            .catch(function () { console.error("Failed to save step."); });
    }

    function renderStepsTree() {
        const tree = document.getElementById("steps-tree");
        tree.innerHTML = "";
        const steps = (draft.steps || []).slice().sort(function (a, b) { return a.step_number - b.step_number; });
        steps.forEach(function (step) {
            const el = document.createElement("div");
            const cls = step.step_number >= 1001 ? "step-inference"
                : step.step_number < 0 ? "step-nok"
                    : step.detect_piece_action ? "step-detect"
                        : step.wait_for_piece_action ? "step-detect"
                            : "step-normal";
            el.className = "step-item " + cls;
            el.innerHTML = `
                <span class="step-num">${step.step_number}</span>
                <span class="step-desc">${step.description || "(no description)"}</span>
                <span class="step-actions">
                    <span class="step-edit" data-num="${step.step_number}" title="Edit step">✏</span>
                    <span class="step-delete" data-num="${step.step_number}" title="Delete step">✕</span>
                </span>
            `;
            el.querySelector(".step-edit").addEventListener("click", function (e) {
                e.stopPropagation();
                editStep(step.step_number);
            });
            el.querySelector(".step-delete").addEventListener("click", function (e) {
                e.stopPropagation();
                removeStep(step.step_number);
            });
            tree.appendChild(el);
        });
    }

    function editStep(stepNum) {
        const step = (draft.steps || []).find(function (s) { return s.step_number === stepNum; });
        if (!step) return;

        // Determine step type from content
        let stepType = "gpio";
        if (step.wait_for_piece_action) {
            stepType = "wait_for_piece";
        } else if (step.detect_piece_action) {
            stepType = "detect_piece";
        } else if (step.camera_action && step.camera_action.length > 0) {
            stepType = "camera";
        } else if (step.step_number >= 1001) {
            stepType = "inference";
        } else if (step.step_number < 0) {
            stepType = "nok";
        }

        // Open modal with step data
        openStepModal(step);

        // Set the step type dropdown
        document.getElementById("step-type").value = stepType;

        // Trigger field rendering
        window.renderStepFields();

        // Populate type-specific fields
        if (stepType === "gpio" || stepType === "nok") {
            if (step.gpio_action && step.gpio_action[0]) {
                document.getElementById("step-gpio-pin").value = step.gpio_action[0].pin_number;
                const actionSelect = document.getElementById("step-gpio-action");
                actionSelect.value = step.gpio_action[0].action;
                // Dispatch change to show/hide timeout field
                actionSelect.dispatchEvent(new Event("change"));
            }
            if (step.delay_after_step) {
                document.getElementById("step-delay").value = step.delay_after_step;
            }
            if (step.wait_for_input_parameters !== undefined) {
                if (step.wait_for_input_parameters.timeout !== undefined) {
                    document.getElementById("step-gpio-timeout").value = step.wait_for_input_parameters.timeout;
                }
                if (step.wait_for_input_parameters.expected_value !== undefined) {
                    document.getElementById("step-gpio-expected").value = step.wait_for_input_parameters.expected_value;
                }
            }
        } else if (stepType === "camera") {
            if (step.camera_action && step.camera_action[0]) {
                document.getElementById("step-cam-port").value = step.camera_action[0].camera_port;
                document.getElementById("step-view-prefix").value = step.camera_action[0].prefix_view || "";
            }
            if (step.delay_after_step) {
                document.getElementById("step-delay").value = step.delay_after_step;
            }
        } else if (stepType === "detect_piece") {
            if (step.detect_piece_action) {
                const da = step.detect_piece_action;
                document.getElementById("step-detect-port").value = da.camera_port || (cameraPorts[0] || "");
                document.getElementById("step-detect-threshold").value = da.darkness_threshold || 80;
                // Show the detection ROI on the corresponding canvas.
                const port = da.camera_port;
                if (port && canvasMap[port] && da.roi) {
                    canvasMap[port].importDetectionRoi(da.roi);
                }
            }
            if (step.delay_after_step) {
                document.getElementById("step-delay").value = step.delay_after_step;
            }
        } else if (stepType === "wait_for_piece") {
            if (step.wait_for_piece_action) {
                const wa = step.wait_for_piece_action;
                document.getElementById("step-wfp-port").value = wa.camera_port || (cameraPorts[0] || "");
                document.getElementById("step-wfp-threshold").value = wa.pixel_diff_threshold ?? 15;
                document.getElementById("step-wfp-poll").value = wa.poll_interval_ms ?? 100;
                document.getElementById("step-wfp-timeout").value = wa.timeout_ms ?? 0;
                document.getElementById("step-wfp-stabilization").value = wa.stabilization_ms ?? 0;
                // Show the detection ROI on the corresponding canvas.
                const port = wa.camera_port;
                if (port && canvasMap[port] && wa.roi) {
                    canvasMap[port].importDetectionRoi(wa.roi);
                }
            }
            if (step.delay_after_step) {
                document.getElementById("step-delay").value = step.delay_after_step;
            }
        } else if (stepType === "inference") {
            if (step.delay_after_step) {
                document.getElementById("step-delay").value = step.delay_after_step;
            }
        }

        // Mark as editing mode - remove old step on save
        document.getElementById("btn-save-step").dataset.editingStep = stepNum;
    }

    function removeStep(stepNum) {
        fetch("/api/builder/remove_step", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ step_number: stepNum }),
        })
            .then(function (r) { return r.json(); })
            .then(function (d) {
                if (d.ok) {
                    draft.steps = d.steps;
                    renderStepsTree();
                    showInferenceReminder();
                }
            });
    }

    function showInferenceReminder() {
        const steps = draft.steps || [];
        const hasInf = steps.some(function (s) { return s.step_number >= 1001; });
        const hasNok = steps.some(function (s) { return s.step_number < 0; });
        const remind = document.getElementById("inference-reminder");
        remind.classList.toggle("hidden", hasInf && hasNok);
    }

    // ── Validate & Save ───────────────────────────────────────────────────

    function bindSaveValidate() {
        document.getElementById("btn-validate").addEventListener("click", function () {
            validate(function (errors) {
                showValidationBanner(errors);
            });
        });

        document.getElementById("btn-save-sequence").addEventListener("click", function () {
            validate(function (errors) {
                if (errors.length > 0) {
                    showValidationBanner(errors);
                    return;
                }
                fetch("/api/builder/save_sequence", { method: "POST" })
                    .then(function (r) { return r.json(); })
                    .then(function (d) {
                        if (d.error) {
                            showValidationBanner([d.error]);
                            return;
                        }
                        window.location.href = d.redirect || "/inspection";
                    })
                    .catch(function () { showValidationBanner(["Server error when saving."]); });
            });
        });
    }

    function validate(callback) {
        fetch("/api/builder/validate", { method: "POST" })
            .then(function (r) { return r.json(); })
            .then(function (d) { callback(d.errors || []); });
    }

    function showValidationBanner(errors) {
        const banner = document.getElementById("validation-banner");
        if (errors.length === 0) {
            banner.classList.add("hidden");
            return;
        }
        banner.innerHTML = "<strong>Fix these issues before saving:</strong><ul>"
            + errors.map(function (e) { return "<li>" + e + "</li>"; }).join("")
            + "</ul>";
        banner.classList.remove("hidden");
        banner.scrollIntoView({ behavior: "smooth" });
    }

    // ── Silent draft save ────────────────────────────────────────────────

    function saveDraftSilently() {
        // Pipelines are already saved via the update_pipeline endpoint.
        // This is a no-op placeholder for section additions which only affect
        // in-memory state until pipelines are explicitly saved.
    }

    function setupSpotlights() {
        const spotlightPins = draft.hardware?.spotlight_gpio_pins || [];
        if (spotlightPins.length === 0) return;

        document.getElementById("spotlights-container").classList.remove("hidden");
        const container = document.getElementById("spotlights-buttons");

        spotlightPins.forEach(function (pin) {
            const btn = document.createElement("button");
            btn.className = "tool-btn";
            btn.textContent = "💡 Pin " + pin + " (OFF)";
            btn.dataset.pin = pin;
            btn.dataset.state = "off";

            btn.addEventListener("click", function () {
                const isOff = btn.dataset.state === "off";
                const newState = isOff ? "on" : "off";

                fetch("/api/builder/toggle_gpio", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ pin: parseInt(pin, 10), state: newState }),
                })
                    .then(r => r.json())
                    .then(d => {
                        if (d.ok) {
                            btn.dataset.state = newState;
                            btn.textContent = "💡 Pin " + pin + (newState === "on" ? " (ON)" : " (OFF)");
                            btn.classList.toggle("active", newState === "on");
                            if (newState === "on") {
                                btn.style.color = "#ff8c00";
                                btn.style.borderColor = "#ff8c00";
                            } else {
                                btn.style.color = "";
                                btn.style.borderColor = "";
                            }
                        }
                    })
                    .catch(function () { console.error("Failed to toggle spotlight GPIO pin."); });
            });

            container.appendChild(btn);
        });

        // Power off if user navigates away while spotlights are on
        window.addEventListener("beforeunload", function () {
            spotlightPins.forEach(function (pin) {
                const btn = document.querySelector(`button[data-pin="${pin}"]`);
                if (btn && btn.dataset.state === "on") {
                    fetch("/api/builder/toggle_gpio", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ pin: parseInt(pin, 10), state: "off" }),
                        keepalive: true, // Attempt to complete request even if page is unloading
                    }).catch(function () { /* ignore errors on unload */ });
                }
            });
        });

    }

    // ── Hardware Settings ───────────────────────────────────────────────────
    function bindHardwareSettings() {
        document.getElementById("btn-hardware-settings").addEventListener("click", openHardwareModal);
        document.getElementById("btn-close-hardware").addEventListener("click", closeHardwareModal);
        document.getElementById("btn-save-hardware").addEventListener("click", saveHardwareSettings);
    }

    // Modal
    function openHardwareModal() {
        const hw  = draft.hardware || {};
        const mod = (IO_CATALOG.modules || []).find(function (m) { return m.name === hw.io_module; }) || null;

        const noGpioNote  = document.getElementById("hw-no-gpio-note");
        const gpioSection = document.getElementById("hw-gpio-section");

        if (mod) {
            noGpioNote.classList.add("hidden");
            gpioSection.classList.remove("hidden");

            const cfg           = hw.gpio_configuration || [];
            const triggerPin    = hw.trigger_input_pin;
            const spotlightPins = hw.spotlight_gpio_pins || [];

            // Pre-fill catalog-default rows with the currently saved values.
            const inputExisting  = {};
            const outputExisting = {};
            cfg.forEach(function (entry) {
                const bcm = entry.pin_number;
                if (entry.type === "input") {
                    inputExisting[bcm] = { description: entry.description || "", is_trigger: bcm === triggerPin };
                } else if (entry.type === "output") {
                    outputExisting[bcm] = {
                        description: entry.description || "",
                        usage: spotlightPins.indexOf(bcm) !== -1 ? "spotlight" : "signal",
                    };
                }
            });

            document.getElementById("hw-input-pins-body").innerHTML  = gpioPinsBuildInputRows(mod, inputExisting);
            document.getElementById("hw-output-pins-body").innerHTML = gpioPinsBuildOutputRows(mod, outputExisting);

            // Restore custom pins wired directly to the Pi, outside the module's
            // default pins (e.g. a hand-wired spotlight not on the IO module).
            const inBcmSet  = mod.gpio.input_pins_bcm;
            const outBcmSet = mod.gpio.output_pins_bcm;
            cfg.forEach(function (entry) {
                const bcm = entry.pin_number;
                if (entry.type === "input" && inBcmSet.indexOf(bcm) === -1) {
                    gpioPinsAddCustomRow("input", "hw-input-pins-body", {
                        pin_number: bcm, description: entry.description || "", is_trigger: bcm === triggerPin,
                    });
                } else if (entry.type === "output" && outBcmSet.indexOf(bcm) === -1) {
                    gpioPinsAddCustomRow("output", "hw-output-pins-body", {
                        pin_number: bcm, description: entry.description || "",
                        usage: spotlightPins.indexOf(bcm) !== -1 ? "spotlight" : "signal",
                    });
                }
            });
        } else {
            noGpioNote.classList.remove("hidden");
            gpioSection.classList.add("hidden");
        }

        const res = hw.camera_capture_resolution || [4608, 2592];
        document.getElementById("hw-capture-w").value = res[0];
        document.getElementById("hw-capture-h").value = res[1];

        document.getElementById("hardware-modal").classList.remove("hidden");
    }

    function closeHardwareModal() {
        document.getElementById("hardware-modal").classList.add("hidden");
    }

    function saveHardwareSettings() {
        const cw = parseInt(document.getElementById("hw-capture-w").value, 10);
        const ch = parseInt(document.getElementById("hw-capture-h").value, 10);

        if (isNaN(cw) || isNaN(ch) || cw < 32 || ch < 32) {
            alert("Capture resolution must be valid.");
            return;
        }

        const payload = { capture_res: [cw, ch] };

        const gpioVisible = !document.getElementById("hw-gpio-section").classList.contains("hidden");
        if (gpioVisible) {
            const gpio = gpioPinsRead("hw-input-pins-body", "hw-output-pins-body");
            if (gpio.triggerPins.length === 0) {
                alert("Mark at least one input pin as trigger (cycle start signal).");
                return;
            }
            payload.gpio_configuration = gpio.gpioConfig;
            payload.trigger_pin        = gpio.triggerPins[0];
            payload.spotlights         = gpio.spotlightPins;
        }

        const btn = document.getElementById("btn-save-hardware");
        btn.textContent = "Saving...";

        fetch("/api/builder/update_hardware", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
            .then(function (r) { return r.json(); })
            .then(function (d) {
                if (d.ok) {
                    draft.hardware = d.hardware;
                    // Update JS environment variables
                    captureW = cw;
                    captureH = ch;
                    Object.keys(canvasMap).forEach(function (ch) {
                        canvasMap[ch].captureW = cw;
                        canvasMap[ch].captureH = ch;
                    });

                    // Force repaint of spotlight buttons in case pins were added or removed
                    const spotlightsContainer = document.getElementById("spotlights-buttons");
                    if (spotlightsContainer) {
                        spotlightsContainer.innerHTML = "";
                        setupSpotlights();
                    }

                    closeHardwareModal();
                } else {
                    alert(d.error || "Error saving hardware configuration.");
                }
            })
            .finally(function () {
                btn.textContent = "Save Settings";
            });
    }

    // ── Initial render ────────────────────────────────────────────────────

    renderStepsTree();
    showInferenceReminder();

}());
