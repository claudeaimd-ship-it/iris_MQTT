/**
 * canvas_tools.js — Fabric.js canvas wrapper for the Iris builder.
 *
 * One CanvasTools instance manages a single Fabric.js canvas bound to a
 * camera card. The builder creates one instance per camera channel.
 *
 * Coordinate system:
 *   All stored parameters use the ORIGINAL capture resolution (e.g. 4608×2592).
 *   The canvas is displayed at a smaller screen size, so every coordinate is
 *   scaled:
 *     real_x = canvas_x * (captureW / displayW)
 *     real_y = canvas_y * (captureH / displayH)
 *
 * Usage:
 *   const ct = new CanvasTools("canvas-A", 4608, 2592);
 *   ct.setBackground(jpegBase64);
 *   ct.setTool("roi");
 *   const pipeline = ct.exportPipeline();  // [{tool, parameters?}, ...]
 */

/* global fabric */

(function (global) {
    "use strict";

    const TOOL_COLORS = {
        roi:           "rgba(29, 111, 216, 0.5)",   // blue — region of interest
        circle:        "rgba(0, 0, 0, 0.85)",        // black mask
        rect:          "rgba(0, 0, 0, 0.85)",        // black mask
        detection_roi: "rgba(255, 140, 0, 0.45)",    // orange — detect_piece_action ROI
    };

    const STROKE_COLORS = {
        roi:           "#1d6fd8",
        circle:        "#333",
        rect:          "#333",
        detection_roi: "#ff8c00",
    };

    /**
     * @param {string} canvasId  — id of the <canvas> element
     * @param {number} captureW  — full capture width in pixels (e.g. 4608)
     * @param {number} captureH  — full capture height in pixels (e.g. 2592)
     */
    function CanvasTools(canvasId, captureW, captureH) {
        this.canvasId   = canvasId;
        this.captureW   = captureW;
        this.captureH   = captureH;
        this.activeTool = "select";
        this._drawing   = false;
        this._origin    = { x: 0, y: 0 };
        this._tmpObject = null;

        this.fc = new fabric.Canvas(canvasId, {
            selection: true,
            backgroundColor: "#0a0a0a",
        });

        this._bindEvents();
    }

    // ── Background image ──────────────────────────────────────────────────

    /**
     * Set the canvas background to a JPEG blob or base64 string.
     * The image is stretched to fill the canvas (caller already set canvas
     * dimensions to match the card size).
     *
     * @param {Blob|string} source  — Blob from fetch or base64 data URL
     */
    CanvasTools.prototype.setBackground = function (source) {
        const self = this;
        const url  = (source instanceof Blob)
            ? URL.createObjectURL(source)
            : source;

        fabric.Image.fromURL(url, function (img) {
            self.fc.setBackgroundImage(img, self.fc.renderAll.bind(self.fc), {
                scaleX: self.fc.width  / img.width,
                scaleY: self.fc.height / img.height,
            });
        });
    };

    /**
     * Resize the Fabric canvas to new pixel dimensions.
     * Call this after the parent card has been laid out by the browser so the
     * canvas fills the card. Must be called before importPipeline so that
     * coordinate scaling uses the correct display size.
     *
     * Shapes are re-imported from their capture-resolution coordinates so they
     * stay visually in the correct position after the resize. The background
     * image is re-scaled to fill the new dimensions.
     *
     * @param {number} width
     * @param {number} height
     */
    CanvasTools.prototype.resize = function (width, height) {
        if (width === this.fc.width && height === this.fc.height) return;

        // Export current shapes in capture-resolution coordinates before the
        // canvas dimensions change (exportPipeline uses fc.width/fc.height).
        var snapshot  = this.exportPipeline(false);
        var detectRoi = this.exportDetectionRoi();

        this.fc.setDimensions({ width: width, height: height });

        // Re-scale the background image to fill the new canvas dimensions.
        // bg.width / bg.height are the natural pixel dimensions of the JPEG.
        var bg = this.fc.backgroundImage;
        if (bg) {
            bg.set({ scaleX: width / bg.width, scaleY: height / bg.height });
        }

        // Remove all drawn shapes and re-import them at the new canvas scale.
        // importPipeline converts from capture-resolution coordinates to the
        // new canvas pixel coordinates automatically.
        this.fc.getObjects().slice().forEach(function (o) { this.fc.remove(o); }, this);
        this.importPipeline(snapshot);
        if (detectRoi) { this.importDetectionRoi(detectRoi); }
        this.fc.renderAll();
    };

    /**
     * Remove the background image (e.g. when no frame has been captured yet).
     */
    CanvasTools.prototype.clearBackground = function () {
        this.fc.setBackgroundImage(null, this.fc.renderAll.bind(this.fc));
    };

    // ── Tool selection ────────────────────────────────────────────────────

    /**
     * Set the active drawing tool.
     * @param {"roi"|"circle"|"rect"|"select"|"delete"} tool
     */
    CanvasTools.prototype.setTool = function (tool) {
        this.activeTool = tool;
        if (tool === "select") {
            this.fc.isDrawingMode = false;
            this.fc.selection     = true;
            this.fc.getObjects().forEach(function (o) { o.selectable = true; });
        } else if (tool === "delete") {
            const active = this.fc.getActiveObjects();
            if (active.length > 0) {
                active.forEach(function (o) { this.fc.remove(o); }, this);
                this.fc.discardActiveObject();
                this.fc.renderAll();
            }
            this.activeTool = "select";
        } else {
            this.fc.isDrawingMode = false;
            this.fc.selection     = false;
            this.fc.getObjects().forEach(function (o) { o.selectable = false; });
        }
    };

    // ── Mouse events ──────────────────────────────────────────────────────

    CanvasTools.prototype._bindEvents = function () {
        const self = this;

        this.fc.on("mouse:down", function (opt) {
            if (self.activeTool === "select" || self.activeTool === "delete") return;
            const pointer = self.fc.getPointer(opt.e);
            self._drawing = true;
            self._origin  = { x: pointer.x, y: pointer.y };

            if (self.activeTool === "roi" || self.activeTool === "rect" || self.activeTool === "detection_roi") {
                self._tmpObject = new fabric.Rect({
                    left:          pointer.x,
                    top:           pointer.y,
                    width:         1,
                    height:        1,
                    fill:          TOOL_COLORS[self.activeTool],
                    stroke:        STROKE_COLORS[self.activeTool],
                    strokeWidth:   2,
                    selectable:    false,
                    evented:       false,
                    objectType:    self.activeTool,   // custom property
                });
            } else if (self.activeTool === "circle") {
                self._tmpObject = new fabric.Ellipse({
                    left:          pointer.x,
                    top:           pointer.y,
                    rx:            1, ry: 1,
                    fill:          TOOL_COLORS.circle,
                    stroke:        STROKE_COLORS.circle,
                    strokeWidth:   2,
                    selectable:    false,
                    evented:       false,
                    objectType:    "circle",
                });
            }

            if (self._tmpObject) {
                self.fc.add(self._tmpObject);
            }
        });

        this.fc.on("mouse:move", function (opt) {
            if (!self._drawing || !self._tmpObject) return;
            const pointer = self.fc.getPointer(opt.e);
            const ox      = self._origin.x;
            const oy      = self._origin.y;

            if (self.activeTool === "roi" || self.activeTool === "rect" || self.activeTool === "detection_roi") {
                const x = Math.min(pointer.x, ox);
                const y = Math.min(pointer.y, oy);
                self._tmpObject.set({
                    left:   x,
                    top:    y,
                    width:  Math.abs(pointer.x - ox),
                    height: Math.abs(pointer.y - oy),
                });
            } else if (self.activeTool === "circle") {
                const rx = Math.abs(pointer.x - ox) / 2;
                const ry = Math.abs(pointer.y - oy) / 2;
                self._tmpObject.set({
                    left: Math.min(pointer.x, ox),
                    top:  Math.min(pointer.y, oy),
                    rx: rx, ry: ry,
                    width:  rx * 2,
                    height: ry * 2,
                });
            }
            self.fc.renderAll();
        });

        this.fc.on("mouse:up", function () {
            if (!self._drawing) return;
            self._drawing = false;
            if (self._tmpObject) {
                self._tmpObject.set({ selectable: true, evented: true });
                self._tmpObject = null;
                self.fc.renderAll();
                // After drawing, switch back to select so the user can move it.
                self.setTool("select");
                if (typeof self.onShapeAdded === "function") self.onShapeAdded();
            }
        });
    };

    // ── Export pipeline ───────────────────────────────────────────────────

    /**
     * Export all drawn shapes as an ordered preprocessing pipeline array.
     *
     * Shapes are exported in Z-order (bottom to top) so ROI crop comes before
     * masks. The caller should append `resize_to_training_resolution` at the end.
     *
     * Coordinates are scaled back to the original capture resolution.
     *
     * @returns {Array<{tool: string, parameters?: object}>}
     */
    CanvasTools.prototype.exportPipeline = function () {
        const pipeline = [];
        const scaleX   = this.captureW / this.fc.width;
        const scaleY   = this.captureH / this.fc.height;

        this.fc.getObjects().forEach(function (obj) {
            const type = obj.objectType;
            if (!type || type === "detection_roi") return;  // detection_roi is not a pipeline tool

            // Use obj.left/top and getScaledWidth/Height() directly instead of
            // getBoundingRect() to avoid the strokeWidth being included in the
            // bounding box, which would cause ~2 px coordinate drift on each
            // "Apply to pipeline" click.
            if (type === "roi" || type === "rect") {
                const x = Math.round(obj.left              * scaleX);
                const y = Math.round(obj.top               * scaleY);
                const w = Math.round(obj.getScaledWidth()  * scaleX);
                const h = Math.round(obj.getScaledHeight() * scaleY);
                const tool = (type === "roi") ? "apply_roi_crop" : "put_black_rectangle";
                pipeline.push({ tool, parameters: { x, y, w, h } });
            } else if (type === "circle") {
                const sw = obj.getScaledWidth();
                const sh = obj.getScaledHeight();
                pipeline.push({
                    tool: "put_black_circle",
                    parameters: {
                        x:      Math.round((obj.left + sw / 2) * scaleX),
                        y:      Math.round((obj.top  + sh / 2) * scaleY),
                        radius: Math.round(Math.max(sw, sh) / 2 * Math.max(scaleX, scaleY)),
                    },
                });
            }
        });

        return pipeline;
    };

    /**
     * Export the detection ROI (detect_piece_action) coordinates.
     * Returns null if no detection_roi object is drawn on this canvas.
     *
     * @returns {{x: number, y: number, w: number, h: number} | null}
     */
    CanvasTools.prototype.exportDetectionRoi = function () {
        const scaleX = this.captureW / this.fc.width;
        const scaleY = this.captureH / this.fc.height;
        const objs   = this.fc.getObjects().filter(function (o) { return o.objectType === "detection_roi"; });
        if (objs.length === 0) return null;
        const obj = objs[0];
        return {
            x: Math.round(obj.left              * scaleX),
            y: Math.round(obj.top               * scaleY),
            w: Math.round(obj.getScaledWidth()  * scaleX),
            h: Math.round(obj.getScaledHeight() * scaleY),
        };
    };

    /**
     * Import a detection ROI onto the canvas (replaces any existing one).
     *
     * @param {{x: number, y: number, w: number, h: number} | null} roi
     */
    CanvasTools.prototype.importDetectionRoi = function (roi) {
        const self = this;
        // Remove existing detection_roi objects.
        this.fc.getObjects()
            .filter(function (o) { return o.objectType === "detection_roi"; })
            .forEach(function (o) { self.fc.remove(o); });
        if (!roi) { this.fc.renderAll(); return; }
        const scaleX = this.fc.width  / this.captureW;
        const scaleY = this.fc.height / this.captureH;
        this.fc.add(new fabric.Rect({
            left:        roi.x * scaleX,
            top:         roi.y * scaleY,
            width:       roi.w * scaleX,
            height:      roi.h * scaleY,
            fill:        TOOL_COLORS.detection_roi,
            stroke:      STROKE_COLORS.detection_roi,
            strokeWidth: 2,
            selectable:  true,
            objectType:  "detection_roi",
        }));
        this.fc.renderAll();
    };

    /**
     * Load a pipeline back onto the canvas (e.g. when restoring from draft).
     * Only geometric tools (roi, put_black_rectangle, put_black_circle) are drawn.
     *
     * @param {Array} pipeline
     */
    CanvasTools.prototype.importPipeline = function (pipeline) {
        const self   = this;
        const scaleX = this.fc.width  / this.captureW;
        const scaleY = this.fc.height / this.captureH;

        (pipeline || []).forEach(function (tool) {
            const p = tool.parameters || {};
            if (tool.tool === "apply_roi_crop") {
                self.fc.add(new fabric.Rect({
                    left:   p.x * scaleX, top:    p.y * scaleY,
                    width:  p.w * scaleX, height: p.h * scaleY,
                    fill:   TOOL_COLORS.roi, stroke: STROKE_COLORS.roi,
                    strokeWidth: 2, selectable: true, objectType: "roi",
                }));
            } else if (tool.tool === "put_black_rectangle") {
                self.fc.add(new fabric.Rect({
                    left:   p.x * scaleX, top:    p.y * scaleY,
                    width:  p.w * scaleX, height: p.h * scaleY,
                    fill:   TOOL_COLORS.rect, stroke: STROKE_COLORS.rect,
                    strokeWidth: 2, selectable: true, objectType: "rect",
                }));
            } else if (tool.tool === "put_black_circle") {
                const rx = (p.radius * scaleX);
                self.fc.add(new fabric.Ellipse({
                    left: (p.x - p.radius) * scaleX, top: (p.y - p.radius) * scaleY,
                    rx: rx, ry: rx * (scaleY / scaleX),
                    fill:   TOOL_COLORS.circle, stroke: STROKE_COLORS.circle,
                    strokeWidth: 2, selectable: true, objectType: "circle",
                }));
            }
        });
        this.fc.renderAll();
    };

    /** Remove all drawn objects (background is preserved). */
    CanvasTools.prototype.clearObjects = function () {
        this.fc.getObjects().slice().forEach(function (o) {
            this.fc.remove(o);
        }, this);
        this.fc.renderAll();
    };

    /** Resize the canvas to match its container element's current size.
     *
     * Re-scales the background image and re-imports all shapes from their
     * capture-resolution coordinates so they remain visually correct after
     * a window resize or browser-zoom change.
     *
     * @param {number} w  — New canvas width in pixels.
     * @param {number} h  — New canvas height in pixels.
     */
    CanvasTools.prototype.resize = function (w, h) {
        // Snapshot current shapes and detection ROI in capture-resolution
        // coordinates BEFORE changing the canvas dimensions.
        var pipeline   = this.exportPipeline();
        var detectRoi  = this.exportDetectionRoi();

        this.fc.setWidth(w);
        this.fc.setHeight(h);

        // Re-scale the background image to fill the new dimensions.
        var bg = this.fc.backgroundImage;
        if (bg) {
            bg.set({ scaleX: w / bg.width, scaleY: h / bg.height });
        }

        // Re-import all shapes at the new canvas scale.
        this.clearObjects();
        this.importPipeline(pipeline);
        if (detectRoi) {
            this.importDetectionRoi(detectRoi);
        }

        this.fc.renderAll();
    };

    global.CanvasTools = CanvasTools;

}(window));
