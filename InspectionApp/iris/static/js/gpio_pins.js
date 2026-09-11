// ── Shared GPIO pin table builder/reader ─────────────────────────────────────
// Used by setup.html (fresh wizard, no existing config) and builder.html
// (Hardware Settings modal, pre-filled from an existing draft/sequence).
//
// A module's default pins come from config/io_module_catalog.json
// (mod.gpio.input_pins_bcm/physical, mod.gpio.output_pins_bcm/physical) and
// are labeled with the module connector port they map to
// (mod.module_connectors.input_ports / output_ports_opto|output_ports),
// index-aligned with the bcm/physical arrays (NOT in IN1→IN4 order — follow
// the catalog's own order, do not assume it).
//
// Besides the module's default pins, operators can add "custom" pins wired
// directly to the Pi (bypassing the IO module, e.g. a hand-wired spotlight),
// identified only by BCM number. Custom rows carry the BCM value in a plain
// number input (`.in-bcm-custom` / `.out-bcm-custom`) instead of the fixed
// `data-bcm` attribute used by catalog rows, so `gpioPinsRead()` can tell
// them apart without any extra bookkeeping.

function gpioPinsBuildInputRows(mod, existingByBcm) {
    existingByBcm = existingByBcm || {};
    const inBcm   = mod.gpio.input_pins_bcm;
    const inPhy   = mod.gpio.input_pins_physical;
    const inPorts = mod.module_connectors.input_ports || [];
    return inBcm.map((bcm, i) => {
        const ex = existingByBcm[bcm] || {};
        return `
        <tr>
            <td><strong>${inPorts[i] || ""}</strong></td>
            <td class="muted">Physical pin ${inPhy[i]}</td>
            <td class="muted">BCM ${bcm}</td>
            <td><input type="text" class="in-desc" data-bcm="${bcm}"
                       placeholder="e.g. PLC start signal" style="width:100%"
                       value="${ex.description || ""}"></td>
            <td style="text-align:center">
                <input type="checkbox" class="in-trigger" data-bcm="${bcm}"
                       title="Use as cycle start trigger" ${ex.is_trigger ? "checked" : ""}>
            </td>
            <td></td>
        </tr>`;
    }).join("");
}

function gpioPinsBuildOutputRows(mod, existingByBcm) {
    existingByBcm = existingByBcm || {};
    const outBcm   = mod.gpio.output_pins_bcm;
    const outPhy   = mod.gpio.output_pins_physical;
    const outPorts = mod.module_connectors.output_ports_opto || mod.module_connectors.output_ports || [];
    return outBcm.map((bcm, i) => {
        const ex    = existingByBcm[bcm] || {};
        const usage = ex.usage || "none";
        return `
        <tr>
            <td><strong>${outPorts[i] || ""}</strong></td>
            <td class="muted">Physical pin ${outPhy[i]}</td>
            <td class="muted">BCM ${bcm}</td>
            <td><input type="text" class="out-desc" data-bcm="${bcm}"
                       placeholder="e.g. OK signal to PLC" style="width:100%"
                       value="${ex.description || ""}"></td>
            <td>
                <select class="out-uso" data-bcm="${bcm}">
                    <option value="none" ${usage === "none" ? "selected" : ""}>Not used</option>
                    <option value="signal" ${usage === "signal" ? "selected" : ""}>PLC Signal (output)</option>
                    <option value="spotlight" ${usage === "spotlight" ? "selected" : ""}>Spotlight</option>
                </select>
            </td>
            <td></td>
        </tr>`;
    }).join("");
}

// Appends one editable custom-pin row (BCM-only, no module/physical pin) to
// the given tbody. `kind` is "input" or "output". `values` (optional) pre-fills
// the row — used by builder.html to restore custom pins already saved in an
// existing sequence (e.g. a hand-wired spotlight on a BCM outside the module).
function gpioPinsAddCustomRow(kind, tbodyId, values) {
    values = values || {};
    const tbody = document.getElementById(tbodyId);
    const row = document.createElement("tr");
    row.className = "custom-pin-row";

    if (kind === "input") {
        row.innerHTML = `
            <td class="muted">Custom</td>
            <td class="muted">—</td>
            <td><input type="number" class="in-bcm-custom" min="0" placeholder="BCM"
                       style="width:70px" value="${values.pin_number != null ? values.pin_number : ""}"></td>
            <td><input type="text" class="in-desc" placeholder="e.g. Custom sensor"
                       style="width:100%" value="${values.description || ""}"></td>
            <td style="text-align:center">
                <input type="checkbox" class="in-trigger" title="Use as cycle start trigger"
                       ${values.is_trigger ? "checked" : ""}>
            </td>
            <td><button type="button" class="btn btn-ghost btn-sm" onclick="this.closest('tr').remove()">✕</button></td>
        `;
    } else {
        const usage = values.usage || "none";
        row.innerHTML = `
            <td class="muted">Custom</td>
            <td class="muted">—</td>
            <td><input type="number" class="out-bcm-custom" min="0" placeholder="BCM"
                       style="width:70px" value="${values.pin_number != null ? values.pin_number : ""}"></td>
            <td><input type="text" class="out-desc" placeholder="e.g. Custom output"
                       style="width:100%" value="${values.description || ""}"></td>
            <td>
                <select class="out-uso">
                    <option value="none" ${usage === "none" ? "selected" : ""}>Not used</option>
                    <option value="signal" ${usage === "signal" ? "selected" : ""}>PLC Signal (output)</option>
                    <option value="spotlight" ${usage === "spotlight" ? "selected" : ""}>Spotlight</option>
                </select>
            </td>
            <td><button type="button" class="btn btn-ghost btn-sm" onclick="this.closest('tr').remove()">✕</button></td>
        `;
    }
    tbody.appendChild(row);
}

// Reads both pin tables (catalog rows + custom rows) back into the payload
// shape expected by /api/setup/finalize and /api/builder/update_hardware.
function gpioPinsRead(inputTbodyId, outputTbodyId) {
    const gpioConfig    = [];
    const triggerPins   = [];
    const spotlightPins = [];

    document.querySelectorAll(`#${inputTbodyId} tr`).forEach(tr => {
        const customBcmEl = tr.querySelector(".in-bcm-custom");
        const bcm = customBcmEl
            ? parseInt(customBcmEl.value)
            : parseInt(tr.querySelector(".in-desc").dataset.bcm);
        if (isNaN(bcm)) return; // empty/incomplete custom row, skip
        const desc      = tr.querySelector(".in-desc").value.trim() || "Input";
        const isTrigger = tr.querySelector(".in-trigger").checked;
        gpioConfig.push({ pin_number: bcm, type: "input", description: desc });
        if (isTrigger) triggerPins.push(bcm);
    });

    document.querySelectorAll(`#${outputTbodyId} tr`).forEach(tr => {
        const customBcmEl = tr.querySelector(".out-bcm-custom");
        const bcm = customBcmEl
            ? parseInt(customBcmEl.value)
            : parseInt(tr.querySelector(".out-desc").dataset.bcm);
        if (isNaN(bcm)) return;
        const desc = tr.querySelector(".out-desc").value.trim() || "Output";
        const uso  = tr.querySelector(".out-uso").value;
        if (uso === "none") return;
        gpioConfig.push({ pin_number: bcm, type: "output", description: desc });
        if (uso === "spotlight") spotlightPins.push(bcm);
    });

    return { gpioConfig, triggerPins, spotlightPins };
}
