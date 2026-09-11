/**
 * calibration.js — Iris Calibration page logic.
 *
 * Handles:
 *  - Step 1: review/relabel gallery (Train OK / Test OK / Test NOK / Discarded)
 *    — capturing new images is done from Inspection's Samples mode instead;
 *    this page only reviews what has already been captured and lets the
 *    operator discard bad captures or restore previously discarded ones.
 *  - Image count polling
 *  - Run Sweep (background task with live progress)
 *  - Run Final Calibration (auto best-block per view — no user input needed)
 *  - Simplified per-view sweep summary with collapsible technical details
 */

(function () {
    "use strict";

    // ── DOM refs ──────────────────────────────────────────────────────────────
    const btnRunSweep       = document.getElementById("btn-run-sweep");
    const btnCancelSweep    = document.getElementById("btn-cancel-sweep");
    const btnRunCalibration = document.getElementById("btn-run-calibration");
    const btnSkipSweep      = document.getElementById("btn-skip-sweep");
    const btnToggleDetails  = document.getElementById("btn-toggle-details");
    const btnCalLoadSeq     = document.getElementById("btn-cal-load-seq");
    const calSeqSelect      = document.getElementById("cal-seq-select");

    const countTrainOk   = document.getElementById("count-train-ok");
    const countTestOk    = document.getElementById("count-test-ok");
    const countTestNok   = document.getElementById("count-test-nok");
    const countDiscarded = document.getElementById("count-discarded");

    // Step 1 — review/relabel gallery
    const reviewSetTabs    = document.getElementById("review-set-tabs");
    const reviewViewSelect = document.getElementById("review-view-select");
    const reviewPageSizeSelect = document.getElementById("review-page-size");
    const reviewGallery    = document.getElementById("review-gallery");
    const reviewPagination = document.getElementById("review-pagination");
    const reviewPageInfo   = document.getElementById("review-page-info");
    const btnReviewPrev    = document.getElementById("btn-review-prev");
    const btnReviewNext    = document.getElementById("btn-review-next");

    // Step 1 review gallery lightbox (enlarged image + move-to-set buttons)
    const reviewLightboxOverlay = document.getElementById("review-lightbox-overlay");
    const reviewLightboxImg     = document.getElementById("review-lightbox-img");
    const reviewLightboxMeta    = document.getElementById("review-lightbox-meta");
    const reviewLightboxActions = document.getElementById("review-lightbox-actions");
    const btnReviewLightboxClose = document.getElementById("btn-review-lightbox-close");
    const btnReviewLightboxPrev  = document.getElementById("btn-review-lightbox-prev");
    const btnReviewLightboxNext  = document.getElementById("btn-review-lightbox-next");

    // Step 1 review gallery: multi-select + delete
    const reviewSelectionBar       = document.getElementById("review-selection-bar");
    const reviewSelectionCount     = document.getElementById("review-selection-count");
    const btnReviewDeleteSelected  = document.getElementById("btn-review-delete-selected");
    const btnReviewCancelSelection = document.getElementById("btn-review-cancel-selection");
    const reviewDeleteModal        = document.getElementById("review-delete-modal");
    const reviewDeleteModalText    = document.getElementById("review-delete-modal-text");
    const btnReviewDeleteCancel      = document.getElementById("btn-review-delete-cancel");
    const btnReviewDeleteModalCancel = document.getElementById("btn-review-delete-modal-cancel");
    const btnReviewDeleteConfirm     = document.getElementById("btn-review-delete-confirm");

    // Production NOK viewer (Step 1 → "View Production NOK…")
    const btnViewProductionNok = document.getElementById("btn-view-production-nok");
    const nokViewerModalOverlay = document.getElementById("nok-viewer-modal-overlay");
    const btnNokViewerClose      = document.getElementById("btn-nok-viewer-close");
    const btnNokViewerDone       = document.getElementById("btn-nok-viewer-done");
    const nokViewerViewSelect   = document.getElementById("nok-viewer-view-select");
    const nokViewerGallery      = document.getElementById("nok-viewer-gallery");
    const nokViewerPagination   = document.getElementById("nok-viewer-pagination");
    const nokViewerPageInfo     = document.getElementById("nok-viewer-page-info");
    const btnNokViewerPrev      = document.getElementById("btn-nok-viewer-prev");
    const btnNokViewerNext      = document.getElementById("btn-nok-viewer-next");

    const nokViewerLightboxOverlay = document.getElementById("nok-viewer-lightbox-overlay");
    const nokViewerLightboxImg     = document.getElementById("nok-viewer-lightbox-img");
    const nokViewerLightboxMeta    = document.getElementById("nok-viewer-lightbox-meta");
    const nokViewerLightboxActions = document.getElementById("nok-viewer-lightbox-actions");
    const btnNokViewerLightboxClose = document.getElementById("btn-nok-viewer-lightbox-close");
    const btnNokViewerLightboxPrev  = document.getElementById("btn-nok-viewer-lightbox-prev");
    const btnNokViewerLightboxNext  = document.getElementById("btn-nok-viewer-lightbox-next");

    const nokViewerSelectionBar       = document.getElementById("nok-viewer-selection-bar");
    const nokViewerSelectionCount     = document.getElementById("nok-viewer-selection-count");
    const btnNokViewerDeleteSelected  = document.getElementById("btn-nok-viewer-delete-selected");
    const btnNokViewerCancelSelection = document.getElementById("btn-nok-viewer-cancel-selection");
    const nokViewerDeleteModal        = document.getElementById("nok-viewer-delete-modal");
    const nokViewerDeleteModalText    = document.getElementById("nok-viewer-delete-modal-text");
    const btnNokViewerDeleteCancel      = document.getElementById("btn-nok-viewer-delete-cancel");
    const btnNokViewerDeleteModalCancel = document.getElementById("btn-nok-viewer-delete-modal-cancel");
    const btnNokViewerDeleteConfirm     = document.getElementById("btn-nok-viewer-delete-confirm");

    const sweepSummary      = document.getElementById("sweep-summary");
    const sweepSummaryList  = document.getElementById("sweep-summary-list");
    const sweepDetailsPanel = document.getElementById("sweep-details-panel");
    const sweepResultsTable = document.getElementById("sweep-results-table");
    const noSweepWarning    = document.getElementById("cal-no-sweep-warning");
    const prevConfigNotice  = document.getElementById("prev-config-notice");
    const prevConfigBlocks  = document.getElementById("prev-config-blocks");

    const progressContainer = document.getElementById("cal-progress-container");
    const progressLabel     = document.getElementById("cal-progress-label");
    const progressFraction  = document.getElementById("cal-progress-fraction");
    const progressBarFill   = document.getElementById("cal-progress-bar-fill");
    const progressMessage   = document.getElementById("cal-progress-message");
    const progressError     = document.getElementById("cal-progress-error");

    // ── State ─────────────────────────────────────────────────────────────────
    let taskRunning      = false;
    let detailsVisible   = false;

    // Step 1 review gallery state
    let reviewSet  = "train_ok";
    let reviewPage = 1;
    let REVIEW_PAGE_SIZE = 20;
    let _reviewItems = [];      // items currently shown on the page (for lightbox nav)
    let _reviewLightboxIdx = 0;
    let _reviewSelected = new Set();  // abs_paths checked for bulk delete (current page only)
    let _pendingDeletePaths = [];     // paths awaiting confirmation in the delete modal

    // Production NOK viewer state
    let nokViewerPage = 1;
    const NOK_VIEWER_PAGE_SIZE = 20;
    let _nokViewerItems = [];         // items currently shown on the page (for lightbox nav)
    let _nokViewerLightboxIdx = 0;
    let _nokViewerSelected = new Set();       // abs_paths checked for bulk delete (current page only)
    let _pendingNokViewerDeletePaths = [];    // paths awaiting confirmation in the delete modal

    // ── Init ──────────────────────────────────────────────────────────────────
    if (btnRunSweep) {
        bindControls();
        pollCounts();
        setInterval(pollCounts, 4000);
        pollStatus();
        setInterval(pollStatus, 2000);
        fetchSweepResults();
        loadReviewGallery();
    }

    // Sequence selector is shown even with no sequence loaded yet, so bind it
    // outside the has_sequence-gated block above.
    if (btnCalLoadSeq) {
        btnCalLoadSeq.addEventListener("click", loadSequence);
    }

    // ── Bind controls ─────────────────────────────────────────────────────────
    function bindControls() {
        btnRunSweep.addEventListener("click",       runSweep);
        btnRunCalibration.addEventListener("click", runCalibration);
        if (btnCancelSweep) {
            btnCancelSweep.addEventListener("click", cancelSweep);
        }
        if (btnSkipSweep) {
            btnSkipSweep.addEventListener("click", skipToCalibrate);
        }
        btnToggleDetails.addEventListener("click",  toggleDetails);

        // Step 1 review/relabel gallery — set tabs, view select, pagination.
        if (reviewSetTabs) {
            reviewSetTabs.querySelectorAll(".cal-tab").forEach(function (tab) {
                tab.addEventListener("click", function () {
                    reviewSet  = tab.dataset.set;
                    reviewPage = 1;
                    reviewSetTabs.querySelectorAll(".cal-tab").forEach(function (t) {
                        t.classList.toggle("cal-tab-active", t === tab);
                    });
                    loadReviewGallery();
                });
            });
        }
        if (reviewViewSelect) {
            reviewViewSelect.addEventListener("change", function () {
                reviewPage = 1;
                loadReviewGallery();
            });
        }
        if (btnReviewPrev) {
            btnReviewPrev.addEventListener("click", function () {
                if (reviewPage > 1) { reviewPage -= 1; loadReviewGallery(); }
            });
        }
        if (btnReviewNext) {
            btnReviewNext.addEventListener("click", function () {
                reviewPage += 1;
                loadReviewGallery();
            });
        }
        if (reviewPageSizeSelect) {
            reviewPageSizeSelect.addEventListener("change", function () {
                REVIEW_PAGE_SIZE = parseInt(reviewPageSizeSelect.value, 10) || 20;
                reviewPage = 1;
                loadReviewGallery();
            });
        }

        // Step 1 review gallery lightbox
        if (btnReviewLightboxClose) {
            btnReviewLightboxClose.addEventListener("click", closeReviewLightbox);
            reviewLightboxOverlay.addEventListener("click", function (e) {
                if (e.target === reviewLightboxOverlay) closeReviewLightbox();
            });
            btnReviewLightboxPrev.addEventListener("click", function () {
                if (_reviewLightboxIdx > 0) openReviewLightbox(_reviewLightboxIdx - 1);
            });
            btnReviewLightboxNext.addEventListener("click", function () {
                if (_reviewLightboxIdx < _reviewItems.length - 1) openReviewLightbox(_reviewLightboxIdx + 1);
            });
            document.addEventListener("keydown", function (e) {
                if (reviewLightboxOverlay.style.display === "none") return;
                if (e.key === "ArrowLeft")  btnReviewLightboxPrev.click();
                if (e.key === "ArrowRight") btnReviewLightboxNext.click();
                if (e.key === "Escape")     closeReviewLightbox();
            });
        }

        // Step 1 review gallery: multi-select + delete
        if (btnReviewDeleteSelected) {
            btnReviewDeleteSelected.addEventListener("click", function () {
                _openDeleteConfirm(Array.from(_reviewSelected));
            });
        }
        if (btnReviewCancelSelection) {
            btnReviewCancelSelection.addEventListener("click", _clearReviewSelection);
        }
        if (btnReviewDeleteCancel)      btnReviewDeleteCancel.addEventListener("click", _closeDeleteConfirm);
        if (btnReviewDeleteModalCancel) btnReviewDeleteModalCancel.addEventListener("click", _closeDeleteConfirm);
        if (btnReviewDeleteConfirm)     btnReviewDeleteConfirm.addEventListener("click", _confirmDelete);
        if (reviewDeleteModal) {
            reviewDeleteModal.addEventListener("click", function (e) {
                if (e.target === reviewDeleteModal) _closeDeleteConfirm();
            });
        }

        // Production NOK viewer (Step 1 → "View Production NOK…")
        if (btnViewProductionNok) {
            btnViewProductionNok.addEventListener("click", openNokViewer);
        }
        if (btnNokViewerClose) {
            btnNokViewerClose.addEventListener("click", closeNokViewer);
            btnNokViewerDone.addEventListener("click", closeNokViewer);
            nokViewerModalOverlay.addEventListener("click", function (e) {
                if (e.target === nokViewerModalOverlay) closeNokViewer();
            });
        }
        if (nokViewerViewSelect) {
            nokViewerViewSelect.addEventListener("change", function () {
                nokViewerPage = 1;
                loadNokViewerGallery();
            });
        }
        if (btnNokViewerPrev) {
            btnNokViewerPrev.addEventListener("click", function () {
                if (nokViewerPage > 1) { nokViewerPage -= 1; loadNokViewerGallery(); }
            });
        }
        if (btnNokViewerNext) {
            btnNokViewerNext.addEventListener("click", function () {
                nokViewerPage += 1;
                loadNokViewerGallery();
            });
        }
        if (btnNokViewerLightboxClose) {
            btnNokViewerLightboxClose.addEventListener("click", closeNokViewerLightbox);
            nokViewerLightboxOverlay.addEventListener("click", function (e) {
                if (e.target === nokViewerLightboxOverlay) closeNokViewerLightbox();
            });
            btnNokViewerLightboxPrev.addEventListener("click", function () {
                if (_nokViewerLightboxIdx > 0) openNokViewerLightbox(_nokViewerLightboxIdx - 1);
            });
            btnNokViewerLightboxNext.addEventListener("click", function () {
                if (_nokViewerLightboxIdx < _nokViewerItems.length - 1) openNokViewerLightbox(_nokViewerLightboxIdx + 1);
            });
            document.addEventListener("keydown", function (e) {
                if (nokViewerLightboxOverlay.style.display === "none") return;
                if (e.key === "ArrowLeft")  btnNokViewerLightboxPrev.click();
                if (e.key === "ArrowRight") btnNokViewerLightboxNext.click();
                if (e.key === "Escape")     closeNokViewerLightbox();
            });
        }
        if (btnNokViewerDeleteSelected) {
            btnNokViewerDeleteSelected.addEventListener("click", function () {
                _openNokViewerDeleteConfirm(Array.from(_nokViewerSelected));
            });
        }
        if (btnNokViewerCancelSelection) {
            btnNokViewerCancelSelection.addEventListener("click", _clearNokViewerSelection);
        }
        if (btnNokViewerDeleteCancel)      btnNokViewerDeleteCancel.addEventListener("click", _closeNokViewerDeleteConfirm);
        if (btnNokViewerDeleteModalCancel) btnNokViewerDeleteModalCancel.addEventListener("click", _closeNokViewerDeleteConfirm);
        if (btnNokViewerDeleteConfirm)     btnNokViewerDeleteConfirm.addEventListener("click", _confirmNokViewerDelete);
        if (nokViewerDeleteModal) {
            nokViewerDeleteModal.addEventListener("click", function (e) {
                if (e.target === nokViewerDeleteModal) _closeNokViewerDeleteConfirm();
            });
        }
    }

    // ── Step 1: review/relabel gallery ─────────────────────────────────────────
    function loadReviewGallery() {
        if (!reviewViewSelect || !reviewGallery) return;
        _clearReviewSelection();
        const viewName = reviewViewSelect.value;
        if (!viewName) {
            reviewGallery.innerHTML = "<p class='muted'>No views configured in the current sequence.</p>";
            reviewPagination.style.display = "none";
            return;
        }
        reviewGallery.innerHTML = "<p class='muted'>Loading…</p>";
        fetch("/api/review/images?set=" + encodeURIComponent(reviewSet)
            + "&view=" + encodeURIComponent(viewName)
            + "&page=" + reviewPage + "&page_size=" + REVIEW_PAGE_SIZE)
        .then(r => r.json())
        .then(d => {
            if (d.error) {
                reviewGallery.innerHTML = "<p class='muted'>" + d.error + "</p>";
                reviewPagination.style.display = "none";
                return;
            }
            renderReviewGallery(d);
        })
        .catch(() => {
            reviewGallery.innerHTML = "<p class='muted'>Could not load images.</p>";
            reviewPagination.style.display = "none";
        });
    }

    function renderReviewGallery(d) {
        const items = d.items || [];
        _reviewItems = items;
        reviewGallery.innerHTML = "";

        if (items.length === 0) {
            reviewGallery.innerHTML = "<p class='muted'>No images.</p>";
        } else {
            items.forEach(function (item, idx) {
                const card = document.createElement("div");
                card.className = "gallery-card";

                const checkbox = document.createElement("input");
                checkbox.type      = "checkbox";
                checkbox.className = "review-card-checkbox";
                checkbox.title     = "Select for delete";
                checkbox.addEventListener("click", function (e) { e.stopPropagation(); });
                checkbox.addEventListener("change", function () {
                    _toggleReviewSelect(item.abs_path, card);
                });
                card.appendChild(checkbox);

                const img = document.createElement("img");
                img.src     = "/api/image_file?path=" + encodeURIComponent(item.abs_path);
                img.alt     = item.filename;
                img.loading = "lazy";
                img.addEventListener("click", function () { openReviewLightbox(idx); });
                card.appendChild(img);

                const meta = document.createElement("div");
                meta.className   = "gallery-card-meta";
                meta.textContent = item.filename;
                card.appendChild(meta);

                const actions = document.createElement("div");
                actions.className = "review-card-actions";
                _reviewMoveTargets(reviewSet).forEach(function (t) {
                    actions.appendChild(_reviewActionButton(t.label, item.abs_path, t.target));
                });
                card.appendChild(actions);

                reviewGallery.appendChild(card);
            });
        }

        const totalPages = Math.ceil(d.total / d.page_size) || 1;
        if (totalPages > 1) {
            reviewPageInfo.textContent    = reviewPage + " / " + totalPages;
            btnReviewPrev.disabled        = reviewPage <= 1;
            btnReviewNext.disabled        = reviewPage >= totalPages;
            reviewPagination.style.display = "flex";
        } else {
            reviewPagination.style.display = "none";
        }
    }

    // Which "move to" buttons to show for a given review set — every set can
    // move to either of the other two active sets, plus Discard; Discarded
    // can only be restored to one of the three active sets.
    function _reviewMoveTargets(set) {
        if (set === "discarded") {
            return [
                { label: "→ Train OK", target: "train_ok" },
                { label: "→ Test OK",  target: "test_ok" },
                { label: "→ Test NOK", target: "test_nok" },
            ];
        }
        const targets = [
            { label: "→ Train OK", target: "train_ok" },
            { label: "→ Test OK",  target: "test_ok" },
            { label: "→ Test NOK", target: "test_nok" },
        ].filter(function (t) { return t.target !== set; });
        targets.push({ label: "Discard", target: "discarded" });
        return targets;
    }

    function _reviewActionButton(label, absPath, targetSet) {
        const btn = document.createElement("button");
        btn.className   = "btn btn-ghost btn-sm";
        btn.textContent = label;
        btn.addEventListener("click", function () { relabelImage(absPath, targetSet); });
        return btn;
    }

    function relabelImage(absPath, targetSet) {
        fetch("/api/review/relabel", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paths: [absPath], target_set: targetSet }),
        })
        .then(r => r.json())
        .then(d => {
            if (d.error) { alert(d.error); return; }
            closeReviewLightbox();
            loadReviewGallery();
            pollCounts();
        })
        .catch(err => alert("Error relabeling image: " + err));
    }

    // ── Step 1 review gallery lightbox ─────────────────────────────────────
    function openReviewLightbox(idx) {
        if (!_reviewItems[idx]) return;
        _reviewLightboxIdx = idx;
        const item = _reviewItems[idx];

        reviewLightboxImg.src  = "/api/image_file?path=" + encodeURIComponent(item.abs_path);
        reviewLightboxMeta.textContent = item.filename;

        reviewLightboxActions.innerHTML = "";
        _reviewMoveTargets(reviewSet).forEach(function (t) {
            reviewLightboxActions.appendChild(_reviewActionButton(t.label, item.abs_path, t.target));
        });
        const deleteBtn = document.createElement("button");
        deleteBtn.className   = "btn btn-danger btn-sm";
        deleteBtn.textContent = "Delete";
        deleteBtn.addEventListener("click", function () { _openDeleteConfirm([item.abs_path]); });
        reviewLightboxActions.appendChild(deleteBtn);

        btnReviewLightboxPrev.disabled = idx <= 0;
        btnReviewLightboxNext.disabled = idx >= _reviewItems.length - 1;
        reviewLightboxOverlay.style.display = "flex";
    }

    function closeReviewLightbox() {
        reviewLightboxOverlay.style.display = "none";
    }

    // ── Step 1 review gallery: multi-select + delete ────────────────────────
    function _toggleReviewSelect(absPath, card) {
        if (_reviewSelected.has(absPath)) {
            _reviewSelected.delete(absPath);
            card.classList.remove("review-card-selected");
        } else {
            _reviewSelected.add(absPath);
            card.classList.add("review-card-selected");
        }
        _updateReviewSelectionBar();
    }

    function _clearReviewSelection() {
        _reviewSelected.clear();
        if (reviewGallery) {
            reviewGallery.querySelectorAll(".review-card-selected").forEach(function (c) {
                c.classList.remove("review-card-selected");
            });
            reviewGallery.querySelectorAll(".review-card-checkbox").forEach(function (cb) {
                cb.checked = false;
            });
        }
        _updateReviewSelectionBar();
    }

    function _updateReviewSelectionBar() {
        if (!reviewSelectionBar) return;
        const n = _reviewSelected.size;
        if (n === 0) {
            reviewSelectionBar.style.display = "none";
            return;
        }
        reviewSelectionBar.style.display = "flex";
        reviewSelectionCount.textContent = n + (n === 1 ? " image selected" : " images selected");
    }

    function _basename(absPath) {
        const parts = absPath.split("/");
        return parts[parts.length - 1];
    }

    function _openDeleteConfirm(paths) {
        if (paths.length === 0) return;
        _pendingDeletePaths = paths;
        reviewDeleteModalText.textContent = paths.length === 1
            ? "Delete image \"" + _basename(paths[0]) + "\" permanently?"
            : "Delete " + paths.length + " images permanently?";
        reviewDeleteModal.style.display = "flex";
    }

    function _closeDeleteConfirm() {
        reviewDeleteModal.style.display = "none";
        _pendingDeletePaths = [];
    }

    function _confirmDelete() {
        if (_pendingDeletePaths.length === 0) return;
        fetch("/api/review/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paths: _pendingDeletePaths }),
        })
        .then(r => r.json())
        .then(d => {
            if (d.error) { alert(d.error); return; }
            _closeDeleteConfirm();
            closeReviewLightbox();
            loadReviewGallery();
            pollCounts();
        })
        .catch(err => alert("Error deleting image(s): " + err));
    }

    // ── Production NOK viewer ────────────────────────────────────────────────
    // Browses NOK inference images for one view across every recorded date
    // (no date picker, unlike the "Recalibrate from Production" modal) and
    // lets the operator delete images no longer needed to free up disk space.
    function openNokViewer() {
        if (!nokViewerModalOverlay || !reviewViewSelect || !nokViewerViewSelect) return;
        // Default to whichever view is currently selected in Step 1's own tabs.
        nokViewerViewSelect.value = reviewViewSelect.value;
        nokViewerPage = 1;
        nokViewerModalOverlay.style.display = "flex";
        loadNokViewerGallery();
    }

    function closeNokViewer() {
        if (nokViewerModalOverlay) nokViewerModalOverlay.style.display = "none";
    }

    function loadNokViewerGallery() {
        if (!nokViewerViewSelect || !nokViewerGallery) return;
        _clearNokViewerSelection();
        const viewName = nokViewerViewSelect.value;
        if (!viewName) {
            nokViewerGallery.innerHTML = "<p class='muted'>No views configured in the current sequence.</p>";
            nokViewerPagination.style.display = "none";
            return;
        }
        nokViewerGallery.innerHTML = "<p class='muted'>Loading…</p>";
        fetch("/api/calibration/nok_images_all?view_name=" + encodeURIComponent(viewName)
            + "&page=" + nokViewerPage + "&page_size=" + NOK_VIEWER_PAGE_SIZE)
        .then(r => r.json())
        .then(d => {
            if (d.error) {
                nokViewerGallery.innerHTML = "<p class='muted'>" + d.error + "</p>";
                nokViewerPagination.style.display = "none";
                return;
            }
            renderNokViewerGallery(d);
        })
        .catch(() => {
            nokViewerGallery.innerHTML = "<p class='muted'>Could not load images.</p>";
            nokViewerPagination.style.display = "none";
        });
    }

    function renderNokViewerGallery(d) {
        const items = d.items || [];
        _nokViewerItems = items;
        nokViewerGallery.innerHTML = "";

        if (items.length === 0) {
            nokViewerGallery.innerHTML = "<p class='muted'>No NOK inference images found for this view.</p>";
        } else {
            items.forEach(function (item, idx) {
                const card = document.createElement("div");
                card.className = "gallery-card";

                const checkbox = document.createElement("input");
                checkbox.type      = "checkbox";
                checkbox.className = "review-card-checkbox";
                checkbox.title     = "Select for delete";
                checkbox.addEventListener("click", function (e) { e.stopPropagation(); });
                checkbox.addEventListener("change", function () {
                    _toggleNokViewerSelect(item.abs_path, card);
                });
                card.appendChild(checkbox);

                const img = document.createElement("img");
                img.src     = "/api/image_file?path=" + encodeURIComponent(item.abs_path);
                img.alt     = item.filename;
                img.loading = "lazy";
                img.addEventListener("click", function () { openNokViewerLightbox(idx); });
                card.appendChild(img);

                const meta = document.createElement("div");
                meta.className   = "gallery-card-meta";
                meta.textContent = item.date_str + " " + item.time_str;
                card.appendChild(meta);

                nokViewerGallery.appendChild(card);
            });
        }

        const totalPages = Math.ceil(d.total / d.page_size) || 1;
        if (totalPages > 1) {
            nokViewerPageInfo.textContent    = nokViewerPage + " / " + totalPages;
            btnNokViewerPrev.disabled        = nokViewerPage <= 1;
            btnNokViewerNext.disabled        = nokViewerPage >= totalPages;
            nokViewerPagination.style.display = "flex";
        } else {
            nokViewerPagination.style.display = "none";
        }
    }

    function openNokViewerLightbox(idx) {
        if (!_nokViewerItems[idx]) return;
        _nokViewerLightboxIdx = idx;
        const item = _nokViewerItems[idx];

        nokViewerLightboxImg.src  = "/api/image_file?path=" + encodeURIComponent(item.abs_path);
        nokViewerLightboxMeta.textContent = item.filename + " — " + item.date_str + " " + item.time_str;

        nokViewerLightboxActions.innerHTML = "";
        const deleteBtn = document.createElement("button");
        deleteBtn.className   = "btn btn-danger btn-sm";
        deleteBtn.textContent = "Delete";
        deleteBtn.addEventListener("click", function () { _openNokViewerDeleteConfirm([item.abs_path]); });
        nokViewerLightboxActions.appendChild(deleteBtn);

        btnNokViewerLightboxPrev.disabled = idx <= 0;
        btnNokViewerLightboxNext.disabled = idx >= _nokViewerItems.length - 1;
        nokViewerLightboxOverlay.style.display = "flex";
    }

    function closeNokViewerLightbox() {
        nokViewerLightboxOverlay.style.display = "none";
    }

    function _toggleNokViewerSelect(absPath, card) {
        if (_nokViewerSelected.has(absPath)) {
            _nokViewerSelected.delete(absPath);
            card.classList.remove("review-card-selected");
        } else {
            _nokViewerSelected.add(absPath);
            card.classList.add("review-card-selected");
        }
        _updateNokViewerSelectionBar();
    }

    function _clearNokViewerSelection() {
        _nokViewerSelected.clear();
        if (nokViewerGallery) {
            nokViewerGallery.querySelectorAll(".review-card-selected").forEach(function (c) {
                c.classList.remove("review-card-selected");
            });
            nokViewerGallery.querySelectorAll(".review-card-checkbox").forEach(function (cb) {
                cb.checked = false;
            });
        }
        _updateNokViewerSelectionBar();
    }

    function _updateNokViewerSelectionBar() {
        if (!nokViewerSelectionBar) return;
        const n = _nokViewerSelected.size;
        if (n === 0) {
            nokViewerSelectionBar.style.display = "none";
            return;
        }
        nokViewerSelectionBar.style.display = "flex";
        nokViewerSelectionCount.textContent = n + (n === 1 ? " image selected" : " images selected");
    }

    function _openNokViewerDeleteConfirm(paths) {
        if (paths.length === 0) return;
        _pendingNokViewerDeletePaths = paths;
        nokViewerDeleteModalText.textContent = paths.length === 1
            ? "Delete image \"" + _basename(paths[0]) + "\" permanently?"
            : "Delete " + paths.length + " images permanently?";
        nokViewerDeleteModal.style.display = "flex";
    }

    function _closeNokViewerDeleteConfirm() {
        nokViewerDeleteModal.style.display = "none";
        _pendingNokViewerDeletePaths = [];
    }

    function _confirmNokViewerDelete() {
        if (_pendingNokViewerDeletePaths.length === 0) return;
        fetch("/api/calibration/delete_nok_images", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paths: _pendingNokViewerDeletePaths }),
        })
        .then(r => r.json())
        .then(d => {
            if (d.error) { alert(d.error); return; }
            _closeNokViewerDeleteConfirm();
            closeNokViewerLightbox();
            loadNokViewerGallery();
        })
        .catch(err => alert("Error deleting image(s): " + err));
    }

    // ── Image counts ──────────────────────────────────────────────────────────
    function pollCounts() {
        fetch("/api/calibration/image_counts")
        .then(r => r.json())
        .then(d => {
            countTrainOk.textContent = d.train_ok  !== undefined ? d.train_ok  : "—";
            countTestOk.textContent  = d.test_ok   !== undefined ? d.test_ok   : "—";
            countTestNok.textContent = d.test_nok  !== undefined ? d.test_nok  : "—";
            if (countDiscarded) {
                countDiscarded.textContent = d.discarded !== undefined ? d.discarded : "—";
            }
        })
        .catch(() => {});
    }

    // ── Skip to Step 3 ───────────────────────────────────────────────────────
    function skipToCalibrate() {
        const section = document.getElementById("section-calibrate");
        if (section) {
            section.scrollIntoView({ behavior: "smooth", block: "start" });
        }
    }

    // ── Cancel sweep ─────────────────────────────────────────────────────────
    function cancelSweep() {
        if (btnCancelSweep) btnCancelSweep.disabled = true;
        fetch("/api/calibration/cancel", { method: "POST" })
        .then(r => r.json())
        .then(() => {
            if (btnCancelSweep) btnCancelSweep.disabled = false;
        })
        .catch(() => {
            if (btnCancelSweep) btnCancelSweep.disabled = false;
        });
    }

    // ── Sweep ─────────────────────────────────────────────────────────────────
    function runSweep() {
        if (taskRunning) { alert("A task is already running. Wait for it to finish."); return; }
        // min_sep_gate is not user-adjustable — always uses the backend default (1.0).
        fetch("/api/calibration/run_sweep", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ params: {} }),
        })
        .then(r => r.json())
        .then(d => {
            if (d.error) { alert(d.error); return; }
            taskRunning = true;
            showProgress("Sweep running…");
            syncTaskButtons();
        })
        .catch(err => alert("Error starting sweep: " + err));
    }

    // ── Calibration ───────────────────────────────────────────────────────────
    function runCalibration() {
        if (taskRunning) { alert("A task is already running. Wait for it to finish."); return; }
        fetch("/api/calibration/run_calibration", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({}),
        })
        .then(r => r.json())
        .then(d => {
            if (d.error) { alert(d.error); return; }
            taskRunning = true;
            showProgress("Calibration running…");
            syncTaskButtons();
        })
        .catch(err => alert("Error starting calibration: " + err));
    }

    // ── Status polling ────────────────────────────────────────────────────────
    function pollStatus() {
        fetch("/api/calibration/status")
        .then(r => r.json())
        .then(d => {
            taskRunning = d.running;

            applyProgress(d.progress || {}, d.mode);
            syncTaskButtons();

            // Enable "Run Calibration" only when sweep has been run or blocks loaded from disk.
            const hasSweep = d.best_blocks && Object.keys(d.best_blocks).length > 0;
            btnRunCalibration.disabled = !hasSweep || taskRunning;
            noSweepWarning.style.display = hasSweep ? "none" : "block";

            // Show previous-config notice in Step 2 when blocks are known.
            if (prevConfigNotice) {
                prevConfigNotice.style.display = hasSweep ? "block" : "none";
                if (hasSweep && prevConfigBlocks) {
                    const labels = Object.entries(d.best_blocks)
                        .map(([view, block]) => `<span class="prev-block-badge">B${block} &mdash; ${view}</span>`)
                        .join("");
                    prevConfigBlocks.innerHTML = labels;
                }
            }
        })
        .catch(() => {});

        fetchSweepResults();
    }

    function syncTaskButtons() {
        btnRunSweep.disabled = taskRunning;
        if (btnCancelSweep) {
            btnCancelSweep.style.display = taskRunning ? "inline-block" : "none";
        }
    }

    // ── Progress UI ───────────────────────────────────────────────────────────
    function showProgress(label) {
        progressLabel.textContent    = label;
        progressFraction.textContent = "";
        progressBarFill.style.width  = "0%";
        progressMessage.textContent  = "";
        progressError.style.display  = "none";
        progressContainer.style.display = "block";
    }

    function applyProgress(p, mode) {
        if (!p || (!p.step && !p.message)) return;
        progressContainer.style.display = "block";

        const step  = p.step  || 0;
        const total = p.total || 1;
        const pct   = total > 0 ? Math.round((step / total) * 100) : 0;

        progressFraction.textContent   = step + " / " + total;
        progressBarFill.style.width    = pct + "%";
        progressMessage.textContent    = p.message || "";

        if (p.done && !p.error) {
            progressLabel.textContent   = "Complete ✓";
            progressBarFill.style.width = "100%";
        } else if (p.done && p.error) {
            progressLabel.textContent   = "Failed ✗";
            progressError.textContent   = p.error;
            progressError.style.display = "block";
        } else if (!progressLabel.textContent) {
            // Fresh page load mid-task: set label based on server mode.
            if (mode === "sweep") {
                progressLabel.textContent = "Sweep running…";
            } else if (mode === "calibration") {
                progressLabel.textContent = "Calibration running…";
            } else {
                progressLabel.textContent = "Running…";
            }
        }
    }

    // ── Sweep summary ─────────────────────────────────────────────────────────
    function fetchSweepResults() {
        fetch("/api/calibration/sweep_results")
        .then(r => r.json())
        .then(d => {
            if (d.sweep_results && d.sweep_results.length > 0) {
                renderSweepSummary(d.sweep_results);
                sweepSummary.style.display = "block";
            }
        })
        .catch(() => {});
    }

    function renderSweepSummary(results) {
        let html = "";
        for (const sweep of results) {
            const sep   = sweep.best_sep_ratio;
            const cls   = sep >= 1.25 ? "sep-good" : sep >= 1.10 ? "sep-warn" : "sep-bad";
            const icon  = sep >= 1.25 ? "✓" : sep >= 1.10 ? "⚠" : "✗";
            const label = sep >= 1.25 ? "Good separation"
                        : sep >= 1.10 ? "Weak separation"
                        : "Poor separation";

            html += `<div class="sweep-summary-row">
                <span class="sweep-view-name">${sweep.view_name}</span>
                <span class="sweep-sep-badge ${cls}">${icon} ${label} — ${sep.toFixed(2)}x</span>
            </div>`;
        }
        sweepSummaryList.innerHTML = html;
        renderDetailTable(results);
    }

    function renderDetailTable(results) {
        let html = "";
        for (const sweep of results) {
            const bestCv = (typeof sweep.best_cv_ok === "number") ? sweep.best_cv_ok.toFixed(4) : "—";
            html += `<div class="sweep-view">
                <h4>${sweep.view_name} — Best: block b${sweep.best_block}
                    (Sep ${sweep.best_sep_ratio.toFixed(3)}x, AUC ${sweep.best_auc.toFixed(4)}, CV ${bestCv})</h4>
                <table class="sweep-table">
                    <thead>
                        <tr>
                            <th>Block</th><th>AUC</th><th>Sep</th>
                            <th>Max OK</th><th>Min NOK</th><th>Std OK</th><th>CV OK</th>
                            <th>Train</th><th>Test OK</th><th>Test NOK</th>
                        </tr>
                    </thead>
                    <tbody>`;
            for (const br of sweep.block_results) {
                const isBest = br.block === sweep.best_block;
                const sc     = br.sep_ratio >= 1.25 ? "sep-good"
                             : br.sep_ratio >= 1.10  ? "sep-warn" : "sep-bad";
                const stdOk  = (typeof br.std_ok === "number") ? br.std_ok.toFixed(5) : "—";
                const cvOk   = (typeof br.cv_ok === "number") ? br.cv_ok.toFixed(4) : "—";
                html += `<tr class="${isBest ? "sweep-best-row" : ""}">
                    <td>${br.block}</td>
                    <td>${br.auc.toFixed(4)}</td>
                    <td><span class="${sc}">${br.sep_ratio.toFixed(3)}x</span></td>
                    <td>${br.max_ok.toFixed(5)}</td>
                    <td>${br.min_nok.toFixed(5)}</td>
                    <td>${stdOk}</td>
                    <td>${cvOk}</td>
                    <td>${br.n_train_ok}</td>
                    <td>${br.n_test_ok}</td>
                    <td>${br.n_test_nok}</td>
                </tr>`;
            }
            html += `</tbody></table></div>`;
        }
        sweepResultsTable.innerHTML = html;
    }

    function toggleDetails() {
        detailsVisible = !detailsVisible;
        sweepDetailsPanel.style.display = detailsVisible ? "block" : "none";
        btnToggleDetails.textContent = detailsVisible
            ? "Hide technical details ▲"
            : "Show technical details ▼";
    }

    // ── Production review modal ───────────────────────────────────────────────
    const reviewOverlay      = document.getElementById("review-modal-overlay");
    const btnOpenReview      = document.getElementById("btn-open-review");
    const btnReviewClose     = document.getElementById("btn-review-close");
    const btnReviewCancel      = document.getElementById("btn-review-cancel");
    const btnReviewAnalyze     = document.getElementById("btn-review-analyze");
    const btnReviewPromote     = document.getElementById("btn-review-promote");
    const btnReviewPromoteOnly = document.getElementById("btn-review-promote-only");
    const btnExportTraceability = document.getElementById("btn-export-traceability");
    const reviewDateInput    = document.getElementById("review-date-input");
    const reviewAnalysisArea = document.getElementById("review-analysis-area");
    const reviewDriftBanner  = document.getElementById("review-drift-banner");
    const reviewSummaryLine  = document.getElementById("review-summary-line");
    const reviewTableBody    = document.getElementById("review-table-body");
    const reviewNoNokMsg     = document.getElementById("review-no-nok-msg");
    const reviewLoadingMsg   = document.getElementById("review-loading-msg");
    const reviewErrorMsg     = document.getElementById("review-error-msg");

    // ── Image gallery modal ───────────────────────────────────────────────────
    const galleryOverlay      = document.getElementById("gallery-modal-overlay");
    const galleryTitle        = document.getElementById("gallery-modal-title");
    const galleryGrid         = document.getElementById("gallery-grid");
    const gallerySummary      = document.getElementById("gallery-summary");
    const galleryLoading      = document.getElementById("gallery-loading");
    const galleryError        = document.getElementById("gallery-error");
    const galleryPagination   = document.getElementById("gallery-pagination");
    const galleryPageLabel    = document.getElementById("gallery-page-label");
    const btnGalleryPrev      = document.getElementById("btn-gallery-prev");
    const btnGalleryNext      = document.getElementById("btn-gallery-next");
    const btnGalleryDone      = document.getElementById("btn-gallery-done");
    const galleryExcludedCount = document.getElementById("gallery-excluded-count");

    // ── Lightbox ──────────────────────────────────────────────────────────────
    const lightboxOverlay  = document.getElementById("lightbox-overlay");
    const lightboxImg      = document.getElementById("lightbox-img");
    const lightboxMeta     = document.getElementById("lightbox-meta");
    const btnLightboxClose = document.getElementById("btn-lightbox-close");
    const btnLightboxPrev  = document.getElementById("btn-lightbox-prev");
    const btnLightboxNext  = document.getElementById("btn-lightbox-next");

    // Gallery runtime state (one active gallery at a time).
    let _galleryState   = null;
    // _reviewExcluded: Map<viewName: string, Set<filename: string>>
    // Persists excluded selections across gallery open/close within the
    // same review modal session. Reset in openReviewModal().
    let _reviewExcluded = new Map();
    /*
     * _galleryState shape:
     * {
     *   apiUrl:      string,
     *   params:      { date, view_name },   // query params for /api/calibration/review_images
     *   pageSize:    number,
     *   currentPage: number,
     *   totalItems:  number,
     *   items:       [{filename, score, time_str}],  // current page
     *   excluded:    Set<string>,           // basenames excluded for this view
     *   lightboxIdx: number,               // index into items for lightbox
     * }
     */

    /**
     * Flatten _reviewExcluded (all views) plus the current open gallery
     * into a single deduplicated array of basenames.
     *
     * @returns {string[]}
     */
    function _getAllExcluded() {
        const all = new Set();
        _reviewExcluded.forEach(function (s) { s.forEach(function (f) { all.add(f); }); });
        if (_galleryState) _galleryState.excluded.forEach(function (f) { all.add(f); });
        return Array.from(all);
    }

    if (btnGalleryDone) {
        btnGalleryDone.addEventListener("click", closeGallery);
        galleryOverlay.addEventListener("click", function (e) {
            if (e.target === galleryOverlay) closeGallery();
        });
        btnGalleryPrev.addEventListener("click", function () {
            if (_galleryState && _galleryState.currentPage > 1)
                loadGalleryPage(_galleryState.currentPage - 1);
        });
        btnGalleryNext.addEventListener("click", function () {
            if (_galleryState) {
                const totalPages = Math.ceil(_galleryState.totalItems / _galleryState.pageSize);
                if (_galleryState.currentPage < totalPages)
                    loadGalleryPage(_galleryState.currentPage + 1);
            }
        });
    }

    if (btnLightboxClose) {
        btnLightboxClose.addEventListener("click", closeLightbox);
        lightboxOverlay.addEventListener("click", function (e) {
            if (e.target === lightboxOverlay) closeLightbox();
        });
        btnLightboxPrev.addEventListener("click", function () {
            if (!_galleryState) return;
            const newIdx = _galleryState.lightboxIdx - 1;
            if (newIdx >= 0) openLightbox(newIdx);
        });
        btnLightboxNext.addEventListener("click", function () {
            if (!_galleryState) return;
            const newIdx = _galleryState.lightboxIdx + 1;
            if (newIdx < _galleryState.items.length) openLightbox(newIdx);
        });
        document.addEventListener("keydown", function (e) {
            if (lightboxOverlay.style.display === "none") return;
            if (e.key === "ArrowLeft")  btnLightboxPrev.click();
            if (e.key === "ArrowRight") btnLightboxNext.click();
            if (e.key === "Escape")     closeLightbox();
        });
    }

    /**
     * Open the image gallery for a specific view's NOK images.
     *
     * @param {string} title      - Modal header text.
     * @param {string} dateStr    - Date in YYYYMMDD format.
     * @param {string} viewName   - View name to load.
     */
    function openImageGallery(title, dateStr, viewName) {
        // Pre-populate excluded from the persistent Map so previous
        // selections are visible immediately when re-opening this view.
        const savedExcluded = _reviewExcluded.get(viewName);
        _galleryState = {
            apiUrl:      "/api/calibration/review_images",
            params:      { date: dateStr, view_name: viewName },
            pageSize:    20,
            currentPage: 1,
            totalItems:  0,
            items:       [],
            excluded:    savedExcluded ? new Set(savedExcluded) : new Set(),
            lightboxIdx: 0,
        };
        galleryTitle.textContent  = title;
        galleryGrid.innerHTML     = "";
        galleryError.style.display    = "none";
        galleryPagination.style.display = "none";
        galleryExcludedCount.textContent = "";
        galleryOverlay.style.display  = "flex";
        loadGalleryPage(1);
    }

    function closeGallery() {
        galleryOverlay.style.display = "none";
        if (_galleryState) {
            // Write back the full Set for this view (supports un-exclude).
            _reviewExcluded.set(
                _galleryState.params.view_name,
                new Set(_galleryState.excluded)
            );
        }
        _galleryState = null;
    }

    function loadGalleryPage(page) {
        if (!_galleryState) return;
        galleryLoading.style.display = "block";
        galleryError.style.display   = "none";
        galleryGrid.innerHTML        = "";
        galleryPagination.style.display = "none";

        const p = _galleryState.params;
        const url = _galleryState.apiUrl
            + "?date=" + encodeURIComponent(p.date)
            + "&view_name=" + encodeURIComponent(p.view_name)
            + "&page=" + page
            + "&page_size=" + _galleryState.pageSize;

        fetch(url)
            .then(function (r) { return r.json(); })
            .then(function (d) {
                galleryLoading.style.display = "none";
                if (d.error) {
                    galleryError.textContent   = d.error;
                    galleryError.style.display = "block";
                    return;
                }
                _galleryState.currentPage = page;
                _galleryState.totalItems  = d.total;
                _galleryState.items       = d.items || [];
                renderGalleryPage();
            })
            .catch(function (err) {
                galleryLoading.style.display = "none";
                galleryError.textContent     = "Request failed: " + err;
                galleryError.style.display   = "block";
            });
    }

    function renderGalleryPage() {
        if (!_galleryState) return;
        const { items, totalItems, currentPage, pageSize, excluded } = _galleryState;
        const totalPages = Math.ceil(totalItems / pageSize) || 1;

        gallerySummary.textContent =
            totalItems + " image" + (totalItems !== 1 ? "s" : "") +
            "  ·  Page " + currentPage + " of " + totalPages;

        galleryGrid.innerHTML = "";

        if (items.length === 0) {
            galleryGrid.innerHTML = "<p class='muted'>No images found.</p>";
        } else {
            items.forEach(function (item, idx) {
                const isExcluded = excluded.has(item.filename);
                const card = document.createElement("div");
                card.className = "gallery-card" + (isExcluded ? " excluded" : "");
                card.dataset.idx = idx;

                const img = document.createElement("img");
                img.src    = "/api/image_file?path=" + encodeURIComponent(
                    // The backend returns just filename; build abs path from state params.
                    // We pass the full path returned by the API when available.
                    item.abs_path || item.filename
                );
                img.alt    = item.filename;
                img.loading = "lazy";
                img.addEventListener("click", function () { openLightbox(idx); });

                const meta = document.createElement("div");
                meta.className = "gallery-card-meta";
                meta.innerHTML =
                    "<span class='gallery-card-score'>" + item.score.toFixed(4) + "</span>" +
                    (item.time_str ? "<br>" + item.time_str : "");

                const exBtn = document.createElement("button");
                exBtn.className = "gallery-card-exclude";
                exBtn.title     = isExcluded ? "Click to include" : "Click to exclude";
                exBtn.textContent = isExcluded ? "\u2713 include" : "\u2717 exclude";
                exBtn.addEventListener("click", function (e) {
                    e.stopPropagation();
                    if (excluded.has(item.filename)) {
                        excluded.delete(item.filename);
                    } else {
                        excluded.add(item.filename);
                    }
                    renderGalleryPage();
                    updateExcludedCount();
                });

                card.appendChild(exBtn);
                card.appendChild(img);
                card.appendChild(meta);
                galleryGrid.appendChild(card);
            });
        }

        // Pagination controls.
        if (totalPages > 1) {
            galleryPageLabel.textContent  = currentPage + " / " + totalPages;
            btnGalleryPrev.disabled       = currentPage <= 1;
            btnGalleryNext.disabled       = currentPage >= totalPages;
            galleryPagination.style.display = "flex";
        } else {
            galleryPagination.style.display = "none";
        }

        updateExcludedCount();
    }

    function updateExcludedCount() {
        if (!_galleryState) return;
        const n = _galleryState.excluded.size;
        galleryExcludedCount.textContent =
            n > 0 ? n + " image" + (n !== 1 ? "s" : "") + " excluded from promotion" : "";
    }

    function openLightbox(idx) {
        if (!_galleryState || !_galleryState.items[idx]) return;
        _galleryState.lightboxIdx = idx;
        const item = _galleryState.items[idx];
        lightboxImg.src = "/api/image_file?path=" + encodeURIComponent(
            item.abs_path || item.filename
        );
        lightboxMeta.textContent =
            "Score: " + item.score.toFixed(4) +
            (item.time_str ? "  \u00b7  " + item.time_str : "");
        btnLightboxPrev.disabled = idx <= 0;
        btnLightboxNext.disabled = idx >= _galleryState.items.length - 1;
        lightboxOverlay.style.display = "flex";
    }

    function closeLightbox() {
        lightboxOverlay.style.display = "none";
    }

    // Latest analysis result stored so the promote action can read it.
    let _reviewAnalysis = null;

    if (btnOpenReview) {
        // Default date = today in YYYY-MM-DD (input[type=date] format).
        const today = new Date();
        const pad   = n => String(n).padStart(2, "0");
        reviewDateInput.value =
            today.getFullYear() + "-" + pad(today.getMonth() + 1) + "-" + pad(today.getDate());

        btnOpenReview.addEventListener("click",    openReviewModal);
        btnReviewClose.addEventListener("click",   closeReviewModal);
        btnReviewCancel.addEventListener("click",  closeReviewModal);
        btnReviewAnalyze.addEventListener("click", runReviewAnalysis);
        btnReviewPromote.addEventListener("click", runPromoteAndRecalibrate);
        btnReviewPromoteOnly.addEventListener("click", runPromoteOnly);
        if (btnExportTraceability) {
            btnExportTraceability.addEventListener("click", function () {
                const dateStr = inputDateToApiDate(reviewDateInput.value);
                if (!dateStr || dateStr.length !== 8) {
                    alert("Please select a valid date.");
                    return;
                }
                window.location.href = "/api/calibration/export_traceability?date=" + dateStr;
            });
        }

        // Close on overlay click (outside the box).
        reviewOverlay.addEventListener("click", function (e) {
            if (e.target === reviewOverlay) closeReviewModal();
        });
    }

    function openReviewModal() {
        _reviewAnalysis  = null;
        _reviewExcluded  = new Map();
        reviewAnalysisArea.style.display = "none";
        reviewLoadingMsg.style.display   = "none";
        reviewErrorMsg.style.display     = "none";
        btnReviewPromote.disabled        = true;
        btnReviewPromoteOnly.disabled    = true;
        if (btnExportTraceability) btnExportTraceability.disabled = true;
        reviewOverlay.style.display      = "flex";
    }

    function closeReviewModal() {
        reviewOverlay.style.display = "none";
    }

    /**
     * Convert "YYYY-MM-DD" (input value) → "YYYYMMDD" (API format).
     */
    function inputDateToApiDate(val) {
        return val.replace(/-/g, "");
    }

    function runReviewAnalysis() {
        const dateStr = inputDateToApiDate(reviewDateInput.value);
        if (!dateStr || dateStr.length !== 8) {
            alert("Please select a valid date.");
            return;
        }

        reviewAnalysisArea.style.display = "none";
        reviewErrorMsg.style.display     = "none";
        reviewLoadingMsg.style.display   = "block";
        btnReviewPromote.disabled        = true;
        btnReviewPromoteOnly.disabled    = true;
        btnReviewAnalyze.disabled        = true;

        fetch("/api/calibration/review_analysis?date=" + dateStr)
        .then(r => r.json())
        .then(d => {
            reviewLoadingMsg.style.display   = "none";
            btnReviewAnalyze.disabled        = false;

            if (d.error) {
                reviewErrorMsg.textContent   = d.error;
                reviewErrorMsg.style.display = "block";
                return;
            }

            _reviewAnalysis = d;
            renderReviewAnalysis(d);
            reviewAnalysisArea.style.display = "block";
            if (btnExportTraceability) btnExportTraceability.disabled = false;
        })
        .catch(err => {
            reviewLoadingMsg.style.display   = "none";
            btnReviewAnalyze.disabled        = false;
            reviewErrorMsg.textContent       = "Request failed: " + err;
            reviewErrorMsg.style.display     = "block";
        });
    }

    function renderReviewAnalysis(d) {
        // ── Summary line ─────────────────────────────────────────────────────
        const nokParts  = d.total_nok_parts || 0;
        const nokGlobal = d.total_parts > 0
            ? " · " + nokParts + " NOK (" + (nokParts / d.total_parts * 100).toFixed(1) + "%)"
            : "";
        reviewSummaryLine.textContent =
            d.total_parts + " parts inspected on " + d.date_str + nokGlobal + ".";

        // ── Global drift banner ──────────────────────────────────────────────
        const nokViews = (d.view_stats || []).filter(vs => vs.nok_count > 0);

        if (d.global_drift_detected) {
            reviewDriftBanner.className     = "review-drift-banner global";
            reviewDriftBanner.textContent   =
                "⚠ Global calibration drift detected: multiple views failed simultaneously. " +
                "These are likely false positives caused by a lighting or exposure change.";
            reviewDriftBanner.style.display = "block";
        } else if (nokViews.length > 0) {
            reviewDriftBanner.className     = "review-drift-banner isolated";
            reviewDriftBanner.textContent   =
                "⚡ Isolated view failures detected. Verify images before promoting — " +
                "these may be real defects.";
            reviewDriftBanner.style.display = "block";
        } else {
            reviewDriftBanner.style.display = "none";
        }

        // ── Table ────────────────────────────────────────────────────────────
        if (nokViews.length === 0) {
            reviewTableBody.innerHTML    = "";
            reviewNoNokMsg.style.display = "block";
            btnReviewPromote.disabled    = true;
            btnReviewPromoteOnly.disabled = true;
            return;
        }

        reviewNoNokMsg.style.display = "none";
        let html = "";

        // Show only views with at least 1 NOK event.
        for (const vs of nokViews) {
            const nokPct   = (vs.nok_rate * 100).toFixed(1) + "%";
            const scoreRng = vs.nok_score_min !== null
                ? vs.nok_score_min.toFixed(2) + " – " + vs.nok_score_max.toFixed(2)
                : "—";

            let badgeClass = "none";
            let badgeText  = "No NOK";
            let badgeTitle = "";
            if (vs.drift_pattern === "global") {
                badgeClass = "global";
                badgeText  = "Global drift";
                badgeTitle = "Failed together with other views — likely false positive";
            } else if (vs.drift_pattern === "isolated") {
                badgeClass = "isolated";
                badgeText  = "Isolated";
                badgeTitle = "Failed independently — verify before promoting";
            }

            const imgsNote = vs.available_images > 0
                ? `<button class="btn btn-ghost btn-sm" onclick="_openGalleryForView('${vs.view_name}')">🔍 ${vs.available_images} NOK</button>`
                : "<span style='color:var(--c-muted)'>none found</span>";

            const canPromote = vs.available_images > 0;

            html += `<tr class="has-nok">
                <td>
                    <input type="checkbox"
                           class="review-view-check"
                           data-view="${vs.view_name}"
                           ${canPromote ? "" : "disabled"}
                           title="${canPromote ? "" : "No images available on disk"}">
                </td>
                <td>${vs.view_name}</td>
                <td>${vs.nok_count} / ${vs.total_parts}</td>
                <td>${nokPct}</td>
                <td>${scoreRng}</td>
                <td>${vs.threshold_max.toFixed(3)}</td>
                <td>${imgsNote}</td>
                <td><span class="drift-badge ${badgeClass}" title="${badgeTitle}">${badgeText}</span></td>
            </tr>`;
        }

        reviewTableBody.innerHTML = html;

        // Enable promote button whenever at least one checkbox is ticked.
        document.querySelectorAll(".review-view-check").forEach(function (cb) {
            cb.addEventListener("change", syncPromoteButton);
        });
        syncPromoteButton();
    }

    // Expose gallery opener to inline onclick handlers in the table.
    window._openGalleryForView = function (viewName) {
        const dateStr = inputDateToApiDate(reviewDateInput.value);
        if (!dateStr) return;
        openImageGallery(
            "NOK Images \u2014 " + viewName + "  \u00b7  " + dateStr,
            dateStr,
            viewName
        );
    };

    function syncPromoteButton() {
        const anyChecked = Array.from(
            document.querySelectorAll(".review-view-check")
        ).some(cb => cb.checked);
        btnReviewPromote.disabled     = !anyChecked;
        btnReviewPromoteOnly.disabled = !anyChecked;
    }

    function runPromoteAndRecalibrate() {
        if (!_reviewAnalysis) return;

        const dateStr = inputDateToApiDate(reviewDateInput.value);
        const viewNames = Array.from(document.querySelectorAll(".review-view-check"))
            .filter(cb => cb.checked)
            .map(cb => cb.dataset.view);

        if (viewNames.length === 0) {
            alert("Select at least one view to promote.");
            return;
        }

        const excludedImages = _getAllExcluded();

        const confirmMsg =
            "Promote " + viewNames.length + " view(s) from " + dateStr + " as training data " +
            "and start re-calibration?\n\n" +
            "Views: " + viewNames.join(", ") +
            (excludedImages.length > 0
                ? "\n\nExcluded images: " + excludedImages.length
                : "") +
            "\n\nThe inspection loop must be stopped before proceeding.";

        if (!window.confirm(confirmMsg)) return;

        btnReviewPromote.disabled     = true;
        btnReviewPromoteOnly.disabled = true;
        btnReviewCancel.disabled      = true;

        fetch("/api/calibration/promote_and_recalibrate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                date_str:         dateStr,
                view_names:       viewNames,
                test_count:       5,
                train_count:      20,
                excluded_images:  excludedImages,
                params:           {},
            }),
        })
        .then(r => r.json())
        .then(d => {
            btnReviewCancel.disabled      = false;
            btnReviewPromoteOnly.disabled = false;
            if (d.error) {
                reviewErrorMsg.textContent   = d.error;
                reviewErrorMsg.style.display = "block";
                btnReviewPromote.disabled    = false;
                return;
            }

            // Show a brief summary of what was promoted.
            let summary = "Images promoted:\n";
            for (const [view, counts] of Object.entries(d.promoted || {})) {
                summary += "  " + view + ": " + counts.train + " → train/OK, " +
                           counts.test + " → test/OK\n";
            }
            summary += "\nRecalibration started. Check the progress bar.";
            alert(summary);

            closeReviewModal();
            taskRunning = true;
            showProgress("Recalibration running…");
            syncTaskButtons();

            // Auto-refresh eval table when recalibration finishes.
            waitForCalibrationDone(loadCalibrationEval);
        })
        .catch(err => {
            btnReviewCancel.disabled      = false;
            btnReviewPromote.disabled     = false;
            btnReviewPromoteOnly.disabled = false;
            reviewErrorMsg.textContent   = "Request failed: " + err;
            reviewErrorMsg.style.display = "block";
        });
    }

    function runPromoteOnly() {
        if (!_reviewAnalysis) return;

        const dateStr   = inputDateToApiDate(reviewDateInput.value);
        const viewNames = Array.from(document.querySelectorAll(".review-view-check"))
            .filter(cb => cb.checked)
            .map(cb => cb.dataset.view);

        if (viewNames.length === 0) {
            alert("Select at least one view to promote.");
            return;
        }

        const excludedImages = _getAllExcluded();

        const confirmMsg =
            "Promote " + viewNames.length + " view(s) from " + dateStr + " as training data?\n\n" +
            "Views: " + viewNames.join(", ") +
            (excludedImages.length > 0
                ? "\n\nExcluded images: " + excludedImages.length
                : "") +
            "\n\nNo recalibration will be triggered.";

        if (!window.confirm(confirmMsg)) return;

        btnReviewPromote.disabled     = true;
        btnReviewPromoteOnly.disabled = true;
        btnReviewCancel.disabled      = true;

        fetch("/api/calibration/promote_only", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                date_str:        dateStr,
                view_names:      viewNames,
                test_count:      5,
                train_count:     20,
                excluded_images: excludedImages,
            }),
        })
        .then(r => r.json())
        .then(d => {
            btnReviewCancel.disabled      = false;
            btnReviewPromote.disabled     = false;
            btnReviewPromoteOnly.disabled = false;
            if (d.error) {
                reviewErrorMsg.textContent   = d.error;
                reviewErrorMsg.style.display = "block";
                return;
            }

            let summary = "Images promoted:\n";
            for (const [view, counts] of Object.entries(d.promoted || {})) {
                summary += "  " + view + ": " + counts.train + " → train/OK, " +
                           counts.test + " → test/OK\n";
            }
            alert(summary);
            closeReviewModal();
        })
        .catch(err => {
            btnReviewCancel.disabled      = false;
            btnReviewPromote.disabled     = false;
            btnReviewPromoteOnly.disabled = false;
            reviewErrorMsg.textContent   = "Request failed: " + err;
            reviewErrorMsg.style.display = "block";
        });
    }

    // =========================================================================
    // Current model evaluation (Step 3)
    // =========================================================================

    const calEvalBlock  = document.getElementById("cal-eval-block");
    const calEvalTbody  = document.getElementById("cal-eval-tbody");
    const calEvalTs     = document.getElementById("cal-eval-ts");
    const calEvalEmpty  = document.getElementById("cal-eval-empty");
    const btnRefreshEval = document.getElementById("btn-refresh-eval");

    if (btnRefreshEval) {
        btnRefreshEval.addEventListener("click", loadCalibrationEval);
    }

    function loadCalibrationEval() {
        fetch("/api/calibration/calibration_eval")
            .then(r => r.json())
            .then(d => renderCalibrationEval(d))
            .catch(() => {});
    }

    function renderCalibrationEval(d) {
        if (!calEvalBlock) return;
        calEvalBlock.style.display = "";

        const rows = d.results || [];
        calEvalTs.textContent = d.timestamp ? "Last calibration: " + d.timestamp : "";

        if (rows.length === 0) {
            calEvalEmpty.style.display  = "";
            calEvalTbody.style.display  = "none";
            document.getElementById("cal-eval-table").style.display = "none";
            return;
        }

        calEvalEmpty.style.display  = "none";
        document.getElementById("cal-eval-table").style.display = "";
        calEvalTbody.style.display  = "";

        calEvalTbody.innerHTML = rows.map(r => {
            const sepClass = r.sep_ratio >= 2.0 ? "eval-good"
                           : r.sep_ratio >= 1.2 ? "eval-warn"
                           : "eval-bad";
            const aucClass = r.auc >= 0.90 ? "eval-good"
                           : r.auc >= 0.70 ? "eval-warn"
                           : "eval-bad";
            return `<tr>
                <td>${r.view_name}</td>
                <td>b${r.block}</td>
                <td class="${sepClass}">${r.sep_ratio.toFixed(3)}×</td>
                <td class="${aucClass}">${r.auc.toFixed(4)}</td>
                <td class="muted">${r.threshold_min.toFixed(4)}</td>
                <td class="muted">${r.threshold_max.toFixed(4)}</td>
            </tr>`;
        }).join("");
    }

    // Poll until calibration progress marks done, then fire callback.
    function waitForCalibrationDone(cb) {
        const poll = setInterval(() => {
            fetch("/api/calibration/progress")
                .then(r => r.json())
                .then(d => {
                    if (!d.running) {
                        clearInterval(poll);
                        cb();
                    }
                })
                .catch(() => clearInterval(poll));
        }, 3000);
    }

    // Auto-load eval table when page loads (shows last stored calibration).
    loadCalibrationEval();

    // Also refresh after a normal Fit Model completes.
    const origBtnCal = document.getElementById("btn-run-calibration");
    if (origBtnCal) {
        origBtnCal.addEventListener("click", () => {
            waitForCalibrationDone(loadCalibrationEval);
        });
    }

    // ── Sequence loading ──────────────────────────────────────────────────────
    // Mirrors the loading-indicator pattern in inspection.js: disable the
    // controls while the hardware reinit request is in flight, then reload
    // the page on success so every sequence-dependent piece of server-rendered
    // state (best_blocks, image counts, view sections) reflects the new
    // sequence — simpler and safer than trying to patch each piece via JS.
    function loadSequence() {
        if (!calSeqSelect || !calSeqSelect.value) return;
        const originalLabel = btnCalLoadSeq.textContent;
        btnCalLoadSeq.disabled  = true;
        calSeqSelect.disabled   = true;
        btnCalLoadSeq.textContent = "Loading…";

        fetch("/api/load_sequence", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: calSeqSelect.value }),
        })
        .then(r => r.json())
        .then(d => {
            if (d.error) {
                showCalError(d.error);
                return;
            }
            window.location.reload();
        })
        .catch(() => {
            showCalError("Could not reach the server to load the sequence.");
        })
        .finally(() => {
            btnCalLoadSeq.disabled  = false;
            calSeqSelect.disabled   = false;
            btnCalLoadSeq.textContent = originalLabel;
        });
    }

    function showCalError(msg) {
        const existing = document.querySelector(".alert-error.inpage");
        if (existing) existing.remove();
        const el = document.createElement("div");
        el.className = "alert alert-error inpage";
        el.textContent = msg;
        document.querySelector(".inspection-controls").after(el);
        setTimeout(() => el.remove(), 5000);
    }

}());
