/**
 * inspection.js — Iris inspection page logic.
 *
 * Handles:
 *   - Mode switching (inference / samples) and label selection.
 *   - Start / stop loop controls.
 *   - On-demand preview capture (loop idle) and MJPEG stream (loop running).
 *   - Last result display with per-view scores and heatmap links.
 *   - Sequence selector and load.
 *   - Status polling every 2 s.
 */

(function () {
    "use strict";

    // ── State ──────────────────────────────────────────────────────────────
    let isRunning      = false;
    let currentMode    = "inference";
    let isDryRun       = false;
    let pollTimer      = null;
    let activeViewTab  = "frames";  // "stream" | "frames" — Live Stream disabled until composite stream is implemented
    let currentPorts   = [];
    let framePollTimer = null;
    var frameObjectUrls  = {};  // blob: URLs per channel — revoked on each update
    let shownStoppedReason = null;  // last stopped_reason/hardware_wedged_message rendered as a critical banner
    let currentLabel   = "ok";     // last known samples-mode label, drives schedule button visibility
    let scheduleStatus = null;    // last known schedule status from /api/status (null when not in samples mode)

    // ── DOM refs ───────────────────────────────────────────────────────────
    const btnStart          = document.getElementById("btn-start");
    const btnStop           = document.getElementById("btn-stop");
    const btnPreview        = document.getElementById("btn-preview");
    const btnDryRun         = document.getElementById("btn-dry-run");
    const btnLoadSeq        = document.getElementById("btn-load-seq");
    const seqSelect         = document.getElementById("seq-select");
    const btnModeInf        = document.getElementById("btn-mode-inference");
    const btnModeSmp        = document.getElementById("btn-mode-samples");
    const labelGroup        = document.getElementById("label-group");
    const resultBadge       = document.getElementById("result-badge");
    const resultBody        = document.getElementById("result-body");
    const gpioEventsHeader  = document.getElementById("gpio-events-header");
    const gpioEventsBody    = document.getElementById("gpio-events-body");
    const cycleCounter      = document.getElementById("cycle-counter");
    const noModelsWarning   = document.getElementById("no-models-warning");
    const streamGrid        = document.getElementById("camera-stream-grid");
    const btnSchedule           = document.getElementById("btn-schedule-captures");
    const scheduleModal         = document.getElementById("schedule-modal");
    const scheduleImagesInput   = document.getElementById("schedule-images-per-window");
    const scheduleHoursInput    = document.getElementById("schedule-interval-hours");
    const scheduleMinutesInput  = document.getElementById("schedule-interval-minutes");
    const scheduleTargetInput   = document.getElementById("schedule-target-images");
    const scheduleTargetNote    = document.getElementById("schedule-target-note");
    const scheduleIntervalTip   = document.getElementById("schedule-interval-tip");
    const btnScheduleConfirm    = document.getElementById("btn-schedule-confirm");
    const btnScheduleDisable    = document.getElementById("btn-schedule-disable");
    const btnScheduleCancel     = document.getElementById("btn-schedule-cancel");

    // ── Initialization ────────────────────────────────────────────────────

    document.addEventListener("DOMContentLoaded", function () {
        bindControls();
        buildCameraGrid([]);   // placeholder; rebuilt when status arrives
        startPolling();
    });

    // ── Status polling ────────────────────────────────────────────────────

    function startPolling() {
        poll();
        pollTimer = setInterval(poll, 2000);
    }

    function poll() {
        fetch("/api/status")
            .then(function (r) { return r.json(); })
            .then(function (d) { applyStatus(d); })
            .catch(function () { /* server unreachable — topbar dot handles it */ });

        if (isRunning) {
            fetch("/api/last_result")
                .then(function (r) { return r.json(); })
                .then(function (d) { if (d) renderResult(d); });
        }
    }

    function applyStatus(d) {
        const wasRunning = isRunning;
        isRunning = d.running;

        // Rebuild camera grid when the sequence (and its camera ports) changes.
        if (d.camera_ports && d.camera_ports.length) buildCameraGrid(d.camera_ports);

        // Per-channel health badge — reflects the last preview attempt, refreshed
        // on every poll so it never goes stale after a reconnect.
        applyChannelStatus(d.channel_status || {});

        // Start / stop button visibility
        btnStart.classList.toggle("hidden",  isRunning);
        btnStop.classList.toggle("hidden",  !isRunning);
        btnPreview.classList.toggle("hidden", isRunning);

        // Mode buttons
        btnModeInf.classList.toggle("active", d.mode === "inference");
        btnModeSmp.classList.toggle("active", d.mode === "samples");
        currentMode = d.mode;

        // Label group visibility
        labelGroup.classList.toggle("hidden", d.mode !== "samples");
        if (d.mode === "samples" && d.label) {
            currentLabel = d.label;
            document.querySelectorAll(".label-btn").forEach(function (b) {
                b.classList.toggle("active", b.dataset.label === d.label);
            });
        }

        scheduleStatus = d.mode === "samples" ? (d.schedule || null) : null;
        updateScheduleButtonVisibility();

        cycleCounter.textContent = "Cycles: " + (d.cycle_count || 0);

        // Dry run toggle — sync from server state, only allow change when stopped
        isDryRun = d.dry_run || false;
        btnDryRun.textContent = isDryRun ? "ON" : "OFF";
        btnDryRun.classList.toggle("active", isDryRun);
        btnDryRun.disabled = isRunning;

        // Force scrap toggle — sync from server state, only allow change when stopped
        forcedScrapActive = d.forced_scrap_cycles > 0;
        updateScrapButtonUI(d.forced_scrap_cycles);


        // Switch from idle to running: start stream
        if (!wasRunning && isRunning) startStream();
        // Switch from running to idle: stop stream
        if (wasRunning && !isRunning) stopStream();

        // Critical error banner — stays visible (not auto-dismissed) until the
        // loop is started again, since it means the loop stopped itself and
        // needs operator attention (e.g. camera failed to reconnect).
        // hardware_wedged_message takes priority — it's the more specific,
        // latched alarm and never auto-clears (only a real service restart does).
        const criticalMsg = d.hardware_wedged_message || d.stopped_reason;
        if (criticalMsg && criticalMsg !== shownStoppedReason) {
            shownStoppedReason = criticalMsg;
            showCriticalBanner(criticalMsg);
        } else if (!criticalMsg && shownStoppedReason) {
            shownStoppedReason = null;
            hideCriticalBanner();
        }
    }

    // ── Camera grid ───────────────────────────────────────────────────────

    function applyChannelStatus(channelStatus) {
        Object.keys(channelStatus).forEach(function (ch) {
            const card = document.getElementById("stream-card-" + ch);
            if (card) card.classList.toggle("disconnected", !!channelStatus[ch]);
        });
    }

    function buildCameraGrid(ports) {
        // Skip rebuild if ports haven't changed — avoids blanking the img elements
        // (and causing flicker) on every status poll.
        if (ports && ports.length) {
            const same = ports.length === currentPorts.length &&
                         ports.every(function (p, i) { return p === currentPorts[i]; });
            if (same) return;
            currentPorts = ports;
        }
        if (!currentPorts.length) return;
        // Revoke stale blob URLs before recreating img elements.
        Object.keys(frameObjectUrls).forEach(function (ch) {
            URL.revokeObjectURL(frameObjectUrls[ch]);
        });
        frameObjectUrls = {};
        streamGrid.innerHTML = "";
        streamGrid.className = "camera-grid cameras-" + (currentPorts.length || 1);
        currentPorts.forEach(function (ch) {
            const card   = document.createElement("div");
            card.className = "camera-card";
            card.id        = "stream-card-" + ch;

            const lbl = document.createElement("span");
            lbl.className = "camera-card-label";
            lbl.textContent = "Camera " + ch;
            card.appendChild(lbl);

            const img = document.createElement("img");
            img.id    = "stream-img-" + ch;
            img.style.cssText = "width:100%;height:100%;object-fit:contain;display:block";
            img.alt   = "Camera " + ch;
            card.appendChild(img);

            streamGrid.appendChild(card);
        });
    }

    // ── MJPEG stream (active only while loop is running) ──────────────────

    function startStream() {
        if (activeViewTab === "frames") {
            startFramePoller();
            return;
        }
        document.querySelectorAll("[id^='stream-img-']").forEach(function (img) {
            img.src = "/stream?" + Date.now();   // cache-bust
        });
    }

    function stopStream() {
        stopFramePoller();
        // Only clear the img elements when on the stream tab.  On the frames
        // tab the images show the last captured frame and must not be blanked.
        if (activeViewTab !== "frames") {
            document.querySelectorAll("[id^='stream-img-']").forEach(function (img) {
                img.src = "";
            });
        }
    }

    // ── Per-channel frame polling (Last Captures tab) ─────────────────────

    function refreshFrames() {
        currentPorts.forEach(function (ch) {
            const img = document.getElementById("stream-img-" + ch);
            if (!img) return;
            fetch("/api/frame/" + ch + "?t=" + Date.now())
                .then(function (r) {
                    if (r.status === 204 || !r.ok) return null;
                    return r.blob();
                })
                .then(function (blob) {
                    if (!blob) return;  // 204 — leave current image unchanged
                    const newUrl = URL.createObjectURL(blob);
                    const oldUrl = frameObjectUrls[ch];
                    frameObjectUrls[ch] = newUrl;
                    img.src = newUrl;
                    // Revoke after the browser has painted the new image.
                    if (oldUrl) URL.revokeObjectURL(oldUrl);
                })
                .catch(function () { /* network error — leave current */ });
        });
    }

    function startFramePoller() {
        stopFramePoller();
        refreshFrames();
        framePollTimer = setInterval(refreshFrames, 2500);
    }

    function stopFramePoller() {
        if (framePollTimer !== null) {
            clearInterval(framePollTimer);
            framePollTimer = null;
        }
    }

    // ── View mode tab switching ───────────────────────────────────────────

    function switchViewTab(tab) {
        // Ignore clicks on disabled tabs.
        const tabBtn = document.querySelector(".view-tab[data-tab='" + tab + "']");
        if (tabBtn && tabBtn.disabled) return;
        activeViewTab = tab;
        document.querySelectorAll(".view-tab").forEach(function (btn) {
            btn.classList.toggle("view-tab-active", btn.dataset.tab === tab);
        });
        if (tab === "stream") {
            stopFramePoller();
            if (isRunning) startStream(); else stopStream();
        } else {
            // Clear MJPEG src to release the connection before polling static frames.
            document.querySelectorAll("[id^='stream-img-']").forEach(function (img) {
                if (img.src.includes("/stream")) img.src = "";
            });
            startFramePoller();
        }
    }

    // ── On-demand preview (loop idle) ─────────────────────────────────────

    btnPreview.addEventListener("click", function () {
        btnPreview.disabled = true;
        btnPreview.textContent = "Capturing…";

        fetch("/api/capture_preview", { method: "POST" })
            .then(function (r) { return r.json(); })
            .then(function (d) {
                const channels = d.captured || [];
                const errors = d.errors || {};
                if (channels.length > 0) buildCameraGrid(channels);
                channels.forEach(function (ch) {
                    const card = document.getElementById("stream-card-" + ch);
                    if (card) card.classList.toggle("disconnected", !!errors[ch]);
                });
                if (d.hardware_wedged_message && d.hardware_wedged_message !== shownStoppedReason) {
                    shownStoppedReason = d.hardware_wedged_message;
                    showCriticalBanner(d.hardware_wedged_message);
                }
                const loads = channels
                    .filter(function (ch) { return !errors[ch]; })
                    .map(function (ch) {
                    return fetch("/api/frame/" + ch + "?t=" + Date.now())
                        .then(function (r) {
                            if (r.status === 204 || !r.ok) return;
                            return r.blob().then(function (blob) {
                                const img = document.getElementById("stream-img-" + ch);
                                if (!img) return;
                                const newUrl = URL.createObjectURL(blob);
                                const oldUrl = frameObjectUrls[ch];
                                frameObjectUrls[ch] = newUrl;
                                img.src = newUrl;
                                if (oldUrl) URL.revokeObjectURL(oldUrl);
                            });
                        })
                        .catch(function () {});
                });
                return Promise.all(loads);
            })
            .finally(function () {
                btnPreview.disabled = false;
                btnPreview.textContent = "📷 Preview";
            });
    });

    // ── Controls ──────────────────────────────────────────────────────────

    function bindControls() {
        btnStart.addEventListener("click", function () {
            apiPost("/api/start")
                .then(function (d) { if (d.error) showError(d.error); else poll(); });
        });

        btnStop.addEventListener("click", function () {
            apiPost("/api/stop")
                .then(function () { poll(); });
        });

        btnDryRun.addEventListener("click", function () {
            if (isRunning) return;   // button is disabled while running; guard anyway
            const newVal = !isDryRun;
            apiPost("/api/set_dry_run", { dry_run: newVal })
                .then(function (d) {
                    if (d.error) showError(d.error);
                    else {
                        isDryRun = newVal;
                        btnDryRun.textContent = isDryRun ? "ON" : "OFF";
                        btnDryRun.classList.toggle("active", isDryRun);
                    }
                });
        });

        btnLoadSeq.addEventListener("click", function () {
            const path = seqSelect.value;
            if (!path) return;
            btnLoadSeq.disabled  = true;
            seqSelect.disabled   = true;
            const originalLabel  = btnLoadSeq.textContent;
            btnLoadSeq.textContent = "Loading…";
            apiPost("/api/load_sequence", { path })
                .then(function (d) {
                    if (d.error) showError(d.error);
                    else {
                        showSuccess("Sequence loaded successfully.");
                        poll();
                    }
                })
                .catch(function () {
                    showError("Could not reach the server to load the sequence.");
                })
                .finally(function () {
                    btnLoadSeq.disabled   = false;
                    seqSelect.disabled    = false;
                    btnLoadSeq.textContent = originalLabel;
                });
        });

        // Mode buttons
        [btnModeInf, btnModeSmp].forEach(function (btn) {
            btn.addEventListener("click", function () {
                const mode = btn.dataset.mode;
                if (mode === currentMode) return;
                apiPost("/api/set_mode", { mode })
                    .then(function (d) {
                        if (d.error) showError(d.error);
                        else {
                            currentMode = mode;
                            btnModeInf.classList.toggle("active", mode === "inference");
                            btnModeSmp.classList.toggle("active", mode === "samples");
                            labelGroup.classList.toggle("hidden", mode !== "samples");
                            updateScheduleButtonVisibility();
                        }
                    });
            });
        });

        // Label buttons
        document.querySelectorAll(".label-btn").forEach(function (btn) {
            btn.addEventListener("click", function () {
                const label = btn.dataset.label;
                apiPost("/api/set_label", { label })
                    .then(function (d) {
                        if (!d.error) {
                            currentLabel = label;
                            document.querySelectorAll(".label-btn").forEach(function (b) {
                                b.classList.toggle("active", b.dataset.label === label);
                            });
                            updateScheduleButtonVisibility();
                        }
                    });
            });
        });

        // View mode tabs
        document.querySelectorAll(".view-tab").forEach(function (btn) {
            btn.addEventListener("click", function () { switchViewTab(btn.dataset.tab); });
        });

        // Schedule Timed Captures
        btnSchedule.addEventListener("click", openScheduleModal);
        btnScheduleCancel.addEventListener("click", closeScheduleModal);

        btnScheduleConfirm.addEventListener("click", function () {
            const body = {
                images_per_window: parseInt(scheduleImagesInput.value, 10),
                interval_minutes: (parseInt(scheduleHoursInput.value, 10) || 0) * 60 +
                                   (parseInt(scheduleMinutesInput.value, 10) || 0),
                target_images: parseInt(scheduleTargetInput.value, 10),
            };
            apiPost("/api/schedule/enable", body)
                .then(function (d) {
                    if (d.error) { showError(d.error); return; }
                    scheduleStatus = d.schedule;
                    updateScheduleButtonVisibility();
                    closeScheduleModal();
                    showSuccess("Schedule Timed Captures enabled.");
                });
        });

        btnScheduleDisable.addEventListener("click", function () {
            apiPost("/api/schedule/disable")
                .then(function () {
                    if (scheduleStatus) scheduleStatus.enabled = false;
                    updateScheduleButtonVisibility();
                    closeScheduleModal();
                    showSuccess("Schedule Timed Captures disabled.");
                });
        });

        scheduleTargetInput.addEventListener("input", updateScheduleTargetConstraints);
    }

    // ── Schedule Timed Captures ────────────────────────────────────────────

    function updateScheduleButtonVisibility() {
        const eligible = currentMode === "samples" && (currentLabel === "ok" || currentLabel === "test_ok");
        btnSchedule.classList.toggle("hidden", !eligible);
        if (!eligible) return;
        const enabled = !!(scheduleStatus && scheduleStatus.enabled);
        btnSchedule.classList.toggle("active", enabled);
        btnSchedule.textContent = enabled ? "🕒 Schedule (ON)" : "🕒 Schedule";
    }

    function openScheduleModal() {
        if (scheduleStatus && scheduleStatus.enabled) {
            scheduleImagesInput.value  = scheduleStatus.images_per_window;
            scheduleHoursInput.value   = Math.floor(scheduleStatus.interval_s / 3600);
            scheduleMinutesInput.value = Math.round((scheduleStatus.interval_s % 3600) / 60);
            scheduleTargetInput.value  = scheduleStatus.target_images;
            btnScheduleDisable.classList.remove("hidden");
        } else {
            btnScheduleDisable.classList.add("hidden");
        }
        updateScheduleTargetConstraints();
        scheduleModal.classList.remove("hidden");
    }

    // Train OK samples only need to cover the normal-part baseline, so the
    // target is hard-capped at 500 — more does not improve calibration, it
    // only spends time and disk space. Test OK samples validate the model
    // against real production, so up to 1000 is fine and going higher is
    // allowed (just slower); a shorter interval also helps there since it
    // captures a wider variety of real production moments over time.
    function updateScheduleTargetConstraints() {
        const isTrainOk = currentLabel === "ok";
        scheduleIntervalTip.style.display = isTrainOk ? "none" : "block";

        if (isTrainOk) {
            scheduleTargetInput.max = "500";
            if (parseInt(scheduleTargetInput.value, 10) > 500) scheduleTargetInput.value = 500;
            scheduleTargetNote.textContent =
                "Train OK only needs to cover the normal-part baseline — 500 images is enough. " +
                "Capturing more does not improve calibration, it only spends extra time and disk space.";
            scheduleTargetNote.style.color = "";
        } else {
            scheduleTargetInput.removeAttribute("max");
            const over1000 = parseInt(scheduleTargetInput.value, 10) > 1000;
            scheduleTargetNote.textContent = over1000
                ? "⚠ Above 1000, capturing will take longer, but the result is more stable and closer to real production."
                : "Up to 1000 Test OK images is a good target to validate the model against real production.";
            scheduleTargetNote.style.color = over1000 ? "var(--c-danger)" : "";
        }
    }

    function closeScheduleModal() {
        scheduleModal.classList.add("hidden");
    }

    // ── Result rendering ──────────────────────────────────────────────────

    function renderResult(result) {
        // Overall badge
        const ok = result.overall_status === "OK";
        resultBadge.textContent  = result.overall_status;
        resultBadge.className    = "result-badge " + (ok ? "ok" : "nok");

        // Meta line
        const metaParts = [
            "Part: " + (result.part_id || "—"),
            "Duration: " + (result.duration_s != null ? result.duration_s + " s" : "—"),
        ];
        if (result.dry_run) metaParts.push("🔵 DRY RUN");
        const meta = document.createElement("p");
        meta.className = "result-meta";
        meta.textContent = metaParts.join("  |  ");

        // Piece detected indicator
        const pieceEl = document.createElement("div");
        pieceEl.className = "result-piece-detected";
        if (result.piece_detected === null || result.piece_detected === undefined) {
            pieceEl.style.display = "none";
        } else {
            const present = result.piece_detected;
            pieceEl.innerHTML = `<span class="piece-label">Piece:</span>
                <span class="piece-badge ${present ? 'ok' : 'nok'}">${present ? '✓ Detected' : '✗ Absent'}</span>`;
        }

        // Failed-channel callout — which camera caused an ERROR_ABORTED cycle, if any.
        let failedChannelEl = null;
        if (result.failed_channel) {
            failedChannelEl = document.createElement("div");
            failedChannelEl.className = "result-failed-channel";
            failedChannelEl.textContent = "⚠ Camera " + result.failed_channel + " failed: " +
                (result.failed_channel_error || "unknown error");
            const card = document.getElementById("stream-card-" + result.failed_channel);
            if (card) card.classList.add("disconnected");
        }

        // View rows
        const rows = (result.view_results || []).map(function (vr) {
            const isOk    = vr.classification === "OK";
            const row     = document.createElement("div");
            row.className = "result-view-row " + (isOk ? "ok" : "nok");
            row.innerHTML = `
                <span class="view-name">${vr.view_name}</span>
                <span class="view-score">${vr.score.toFixed(4)} [${vr.threshold_min.toFixed(4)}–${vr.threshold_max.toFixed(4)}]</span>
                <span class="view-heatmap" data-view="${vr.view_name}">heatmap</span>
            `;
            row.querySelector(".view-heatmap").addEventListener("click", function () {
                openHeatmap(vr.view_name);
            });
            return row;
        });

        resultBody.innerHTML = "";
        resultBody.appendChild(meta);
        if (failedChannelEl) resultBody.appendChild(failedChannelEl);
        resultBody.appendChild(pieceEl);
        rows.forEach(function (r) { resultBody.appendChild(r); });

        // GPIO event panel
        const triggers = result.triggers || [];
        gpioEventsHeader.style.display = triggers.length > 0 ? "flex" : "none";
        gpioEventsBody.innerHTML = "";
        triggers.forEach(function (t) {
            const el = document.createElement("div");
            el.className = "gpio-event-row";
            const dirIcon   = t.direction === "INPUT" ? "←" : "→";
            const resultCls = t.result === "OK" ? "ok" : t.result === "TIMEOUT" ? "timeout" : "sent";
            el.innerHTML = `
                <span class="gpio-dir">${dirIcon}</span>
                <span class="gpio-pin">BCM ${t.pin}</span>
                <span class="gpio-action">${t.action}</span>
                <span class="gpio-result ${resultCls}">${t.result}</span>
            `;
            gpioEventsBody.appendChild(el);
        });
    }

    function openHeatmap(viewName) {
        const url = "/api/heatmap/" + encodeURIComponent(viewName);
        const w = window.open("", "_blank", "width=1024,height=768,resizable=yes");
        if (!w) return;
        w.document.write(
            "<!DOCTYPE html><html><head><title>Heatmap \u2014 " + viewName + "</title>" +
            "<style>body{margin:0;background:#111;display:flex;align-items:center;" +
            "justify-content:center;min-height:100vh}" +
            "img{max-width:100%;max-height:100vh;object-fit:contain}</style></head>" +
            "<body><img src='" + url + "' alt='heatmap'></body></html>"
        );
        w.document.close();
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    function apiPost(url, body) {
        return fetch(url, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    body ? JSON.stringify(body) : "{}",
        }).then(function (r) { return r.json(); });
    }

    function showError(msg) {
        const existing = document.querySelector(".alert-error.inpage");
        if (existing) existing.remove();
        const el       = document.createElement("div");
        el.className   = "alert alert-error inpage";
        el.textContent = msg;
        document.querySelector(".inspection-controls").after(el);
        setTimeout(function () { el.remove(); }, 5000);
    }

    function showSuccess(msg) {
        const existing = document.querySelector(".alert-success.inpage");
        if (existing) existing.remove();
        const el       = document.createElement("div");
        el.className   = "alert alert-success inpage";
        el.textContent = msg;
        document.querySelector(".inspection-controls").after(el);
        setTimeout(function () { el.remove(); }, 4000);
    }

    function showCriticalBanner(msg) {
        hideCriticalBanner();
        const el       = document.createElement("div");
        el.id          = "critical-error-banner";
        el.className   = "alert alert-error inpage";
        el.textContent = "⚠ " + msg;
        document.querySelector(".inspection-controls").after(el);
        // No setTimeout — stays until the operator restarts the loop.
    }

    function hideCriticalBanner() {
        const el = document.getElementById("critical-error-banner");
        if (el) el.remove();
    }

    // ── Forced SCRAP Mode ──────────────────────────────────────────────────────────
    
    const btnForcedScrap = document.getElementById("btn-forced-scrap");
    let forcedScrapActive = false;

    btnForcedScrap.addEventListener("click", function () {
        const cycles = forcedScrapActive ? 0 : 3;  // Force next 3 cycles into scrap mode
        apiPost("/api/set_forced_scrap", { cycles: cycles })
            .then(function (d) {
                if (d.ok) {
                    forcedScrapActive = (d.forced_scrap_cycles > 0);
                    updateScrapButtonUI(d.forced_scrap_cycles);
                }
            });
    });
    
    function updateScrapButtonUI(cycles) {
        if (cycles > 0) {
            btnForcedScrap.classList.add("active");
            btnForcedScrap.textContent = "ON (" + cycles + " cycles left)";
        } else {
            btnForcedScrap.classList.remove("active");
            btnForcedScrap.textContent = "OFF";
        }
    }

}());

    