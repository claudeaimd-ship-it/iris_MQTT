import os
import json
import cv2
import numpy as np
from datetime import datetime

from app.src.interfaces.IRepository import IRepository
from app.src.core.models.Part import Part


class LocalStorageAdapter(IRepository):
    """
    Persists inspection results to the local filesystem.

    Writes two artifacts for every inspected part:

    1. **Traceability log** (JSON Lines, ``.jsonl``): one JSON object per line
       appended to a daily file under
       ``{traceability_path}/YYYYMM/YYYYMMDD_results.jsonl``.

       Each record contains three normalized sections that map directly to
       database table rows when the data is later imported:

       - ``part``: scalar fields for the ``inspection_parts`` table.
       - ``view_results``: list of per-view dicts for ``inspection_view_results``.
       - ``trigger_events``: list of per-trigger dicts for ``inspection_trigger_events``.

    2. **Captured images** (optional): saves raw captured frames to
       ``{inference_images_path}/YYYYMMDD/{part_id}/{view_name}.jpg``.

    Attributes:
        _traceability_path (str): Base directory for the traceability log files.
        _inference_images_path (str | None): Base directory for saving captured
            images. If ``None``, images are not saved.
    """

    def __init__(self, traceability_path: str, inference_images_path: str | None = None):
        """
        Args:
            traceability_path (str): Base directory for traceability ``.jsonl`` files.
                Created automatically if it does not exist.
            inference_images_path (str | None): Base directory for saving captured
                images alongside results. Pass ``None`` to skip image saving.
        """
        self._traceability_path     = traceability_path
        self._inference_images_path = inference_images_path
        os.makedirs(traceability_path, exist_ok=True)

    # =========================================================================
    # IRepository interface
    # =========================================================================

    def save_inspection_result(self, part: Part) -> bool:
        """
        Persist the inspection results for a completed part.

        Args:
            part (Part): Completed Part with ``inspection_results``, ``triggers``,
                ``date_inspected``, and ``time_inspected`` populated.

        Returns:
            bool: ``True`` if the write succeeded, ``False`` on any error.
        """
        try:
            self._write_traceability_record(part)
            return True
        except Exception as e:
            print(f"[ERROR] LocalStorageAdapter: failed to save results for '{part.part_id}': {e}")
            return False

    def get_inspection_result(self, part_id: str) -> Part | None:
        """
        Retrieve a previously saved Part.

        The traceability log is append-only JSON Lines; full reconstruction of a
        Part object from it is not implemented. This method exists to satisfy
        the IRepository contract and will be replaced when a database adapter is added.

        Args:
            part_id (str): Unique identifier for the part.

        Returns:
            None
        """
        print("[WARN] LocalStorageAdapter.get_inspection_result: retrieval not implemented.")
        return None

    # =========================================================================
    # Additional public methods (outside IRepository contract)
    # =========================================================================

    def save_frames(self, part_id: str, captured_frames: dict[str, np.ndarray]) -> bool:
        """
        Save the captured raw frames to the inference images directory.

        Files are written to::

            {inference_images_path}/YYYYMM/YYYYMMDD/YYYYMMDDHHmmss_{part_id}_{view_name}.jpg

        The timestamp prefix is the wall-clock time at save, which is effectively
        the end of the inspection cycle. Multiple views saved in the same call share
        the same timestamp prefix. The ``part_id`` is embedded in the filename so
        ``TraceabilityReviewService`` can match an image to its JSONL record by
        exact ``part_id`` instead of a time-window guess — the wait for a physical
        piece can make the gap between ``part.date_inspected`` (cycle start) and
        this save timestamp (cycle end) arbitrarily long, which made window-based
        matching unreliable.

        Args:
            part_id (str): Unique identifier for the part. Embedded in each
                filename and used for error logging.
            captured_frames (dict[str, np.ndarray]): Frames keyed by view_name.

        Returns:
            bool: ``True`` if all frames were saved, ``False`` on any error.
        """
        if self._inference_images_path is None:
            return True  # Image saving is disabled.

        now        = datetime.now()
        month_dir  = now.strftime("%Y%m")
        day_dir    = now.strftime("%Y%m%d")
        ts_prefix  = now.strftime("%Y%m%d%H%M%S")
        output_dir = os.path.join(self._inference_images_path, month_dir, day_dir)
        os.makedirs(output_dir, exist_ok=True)

        success = True
        for view_name, frame in captured_frames.items():
            file_path = os.path.join(output_dir, f"{ts_prefix}_{part_id}_{view_name}.jpg")
            try:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, bgr)
            except Exception as e:
                print(f"[ERROR] LocalStorageAdapter: could not save frame '{view_name}' "
                      f"for part '{part_id}': {e}")
                success = False

        return success

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _write_traceability_record(self, part: Part) -> None:
        """
        Append one JSON record to the daily traceability log file.

        The record structure maps directly to three database tables:

        .. code-block:: json

            {
                "part": {
                    "part_id": "...",
                    "date_inspected": "YYYYMMDD_HHMMSS",
                    "duration_s": 3.1415,
                    "overall_status": "OK",
                    "piece_detected": null
                },
                "view_results": [
                    {
                        "view_name": "front_view_section_1_A",
                        "classification": "OK",
                        "score": 0.000412,
                        "threshold_min": 0.0001,
                        "threshold_max": 0.005
                    }
                ],
                "trigger_events": [
                    {
                        "step_number": 1,
                        "direction": "INPUT",
                        "pin": 13,
                        "action": "wait_for_input",
                        "result": "OK"
                    }
                ]
            }
        """
        date_str  = part.date_inspected.strftime("%Y%m%d_%H%M%S")
        month_str = part.date_inspected.strftime("%Y%m")
        day_str   = part.date_inspected.strftime("%Y%m%d")

        log_dir  = os.path.join(self._traceability_path, month_str)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{day_str}_results.jsonl")

        record = {
            "part": {
                "part_id":        part.part_id,
                "model_id":       part.model_id,
                "date_inspected": date_str,
                "duration_s":     round(part.time_inspected, 4) if part.time_inspected is not None else None,
                "overall_status": "ERROR_ABORTED" if getattr(part, "system_error_paused", False) else (
                    ("OK" if part.overall_status else "NOK") + ("_SCRAP" if part.forced_scrap else "") + ("_DR" if part.dry_run else "")
                ),
                "piece_detected": part.piece_detected,
            },
            "view_results": [
                {
                    "view_name":      result.view,
                    "classification": "OK" if result.is_ok else "NOK",
                    "score":          round(result.score, 6),
                    "threshold_min":  round(result.threshold_used[0], 6),
                    "threshold_max":  round(result.threshold_used[1], 6),
                }
                for result in part.inspection_results
            ],
            "trigger_events": [
                {
                    "step_number": trig.step_number,
                    "direction":   trig.direction,
                    "pin":         trig.pin,
                    "action":      trig.action,
                    "result":      trig.result,
                }
                for trig in part.triggers
            ],
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[OK] Traceability record written: '{log_path}'")

