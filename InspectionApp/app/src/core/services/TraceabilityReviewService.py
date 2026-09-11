import json
import os
import re
import shutil
from datetime import datetime, timedelta

from app.src.core.models.TraceabilityReviewModels import ReviewAnalysis, ViewReviewStats

# Views that failed simultaneously in the same part cycle are considered a
# "global drift" event.  Threshold: at least this fraction of all views in a
# cycle must be NOK for the cycle to count as a global-drift cycle.
_GLOBAL_DRIFT_FRACTION = 0.6

# Inference image filenames written by LocalStorageAdapter.save_frames() as of
# the part_id-embedding fix: "{YYYYMMDDHHMMSS}_{part_id}_{view_name}.jpg".
# part_id itself has the form "YYYYMMDD-NNNN" (see GuiInferenceAdapter._generate_part_id).
# Images saved before this fix (bare "{ts}_{view_name}.jpg") simply won't match
# any group and are silently skipped — by design, since the whole point is to
# stop guessing from timestamps.
_IMAGE_FILENAME_RE = re.compile(r"^\d{14}_(\d{8}-\d+)_(.+)\.jpg$", re.IGNORECASE)


class TraceabilityReviewService:
    """Analyzes production traceability data and promotes NOK images to training.

    Reads ``YYYYMMDD_results.jsonl`` files to compute per-view NOK statistics,
    detects systemic calibration drift, and copies the corresponding inference
    images to the training/test directories for re-calibration.

    This service is a pure core service; it performs direct filesystem access
    (consistent with ``CalibrationService``).

    Attributes:
        _traceability_path: Base directory that contains ``YYYYMM/`` sub-folders
            with daily ``.jsonl`` files.
        _inference_images_path: Base directory that contains ``YYYYMM/YYYYMMDD/``
            sub-folders with per-part inference images.
        _images_path: Base directory for training images
            (``{images_path}/train/OK/{view}`` etc.).
    """

    def __init__(
        self,
        traceability_path: str,
        inference_images_path: str,
        images_path: str,
    ) -> None:
        """
        Args:
            traceability_path: Base directory for traceability ``.jsonl`` files.
            inference_images_path: Base directory for saved inference images.
            images_path: Base directory for training/test images.
        """
        self._traceability_path    = traceability_path
        self._inference_images_path = inference_images_path
        self._images_path          = images_path

    # =========================================================================
    # Public API
    # =========================================================================

    def resolve_jsonl_path(self, date_str: str) -> str:
        """Return the expected path to the JSONL traceability file for a date.

        Args:
            date_str: Date in ``YYYYMMDD`` format.

        Returns:
            str: Absolute path (may or may not exist on disk).
        """
        return self._jsonl_path_for_date(date_str)

    def analyze(self, date_str: str) -> ReviewAnalysis:
        """Analyze all inspection records for a given date.

        Reads the daily JSONL file, computes per-view NOK rates and score
        statistics, detects global calibration drift patterns, and counts
        available inference images on disk.

        Args:
            date_str: Date to analyze in ``YYYYMMDD`` format (e.g. ``"20260616"``).

        Returns:
            ReviewAnalysis: Full analysis result.  ``view_stats`` is empty if
            no records with complete view results are found for that date.
        """
        jsonl_path = self._jsonl_path_for_date(date_str)
        records    = self._read_jsonl(jsonl_path)

        # Keep only records that have view_results (skip ERROR_ABORTED).
        valid = [r for r in records if r.get("view_results")]

        # A part is NOK when at least one of its view results is NOK.
        total_nok_parts = sum(
            1 for r in valid
            if any(vr.get("classification") == "NOK" for vr in r["view_results"])
        )

        analysis = ReviewAnalysis(
            date_str=date_str,
            total_parts=len(valid),
            total_nok_parts=total_nok_parts,
            jsonl_path=jsonl_path,
        )

        if not valid:
            return analysis

        # ── Per-view aggregation ──────────────────────────────────────────────
        view_data: dict[str, dict] = {}
        for record in valid:
            for vr in record["view_results"]:
                vname = vr["view_name"]
                if vname not in view_data:
                    view_data[vname] = {
                        "all_scores":  [],
                        "nok_scores":  [],
                        "nok_count":   0,
                        "threshold_min": vr.get("threshold_min", 0.0),
                        "threshold_max": vr.get("threshold_max", 0.0),
                    }
                score = vr.get("score", 0.0)
                is_nok = vr.get("classification") == "NOK"
                view_data[vname]["all_scores"].append(score)
                view_data[vname]["threshold_min"] = vr.get("threshold_min", 0.0)
                view_data[vname]["threshold_max"] = vr.get("threshold_max", 0.0)
                if is_nok:
                    view_data[vname]["nok_count"]  += 1
                    view_data[vname]["nok_scores"].append(score)

        # ── Global drift detection ────────────────────────────────────────────
        # A cycle is a "global drift cycle" when ≥ _GLOBAL_DRIFT_FRACTION of
        # all views in that cycle classified as NOK.
        global_drift_part_ids: set[str] = set()
        for record in valid:
            vrs      = record["view_results"]
            n_views  = len(vrs)
            n_nok    = sum(1 for vr in vrs if vr.get("classification") == "NOK")
            if n_views > 0 and n_nok / n_views >= _GLOBAL_DRIFT_FRACTION:
                global_drift_part_ids.add(record["part"]["part_id"])

        analysis.global_drift_detected = len(global_drift_part_ids) > 0

        # ── Per-view drift pattern ────────────────────────────────────────────
        # Count how many of each view's NOK events come from global-drift cycles.
        global_nok_per_view: dict[str, int] = {}
        for record in valid:
            if record["part"]["part_id"] in global_drift_part_ids:
                for vr in record["view_results"]:
                    if vr.get("classification") == "NOK":
                        vname = vr["view_name"]
                        global_nok_per_view[vname] = global_nok_per_view.get(vname, 0) + 1

        # ── Build image index for the day ─────────────────────────────────────
        image_index = self._build_image_index(date_str)

        # ── Assemble ViewReviewStats ──────────────────────────────────────────
        for vname, data in view_data.items():
            nok_count    = data["nok_count"]
            total_for_view = len(data["all_scores"])
            nok_rate     = nok_count / total_for_view if total_for_view > 0 else 0.0

            # Drift pattern classification.
            if nok_count == 0:
                drift_pattern = "none"
            else:
                global_nok = global_nok_per_view.get(vname, 0)
                # If ≥ 70 % of this view's NOK events were in global-drift
                # cycles, classify as global; otherwise isolated.
                if global_nok / nok_count >= 0.70:
                    drift_pattern = "global"
                else:
                    drift_pattern = "isolated"

            # Count images available on disk for this view's NOK events.
            available_images = self._count_matching_images(
                valid, vname, image_index, date_str
            )

            analysis.view_stats.append(ViewReviewStats(
                view_name=vname,
                total_parts=total_for_view,
                nok_count=nok_count,
                nok_rate=nok_rate,
                all_scores=data["all_scores"],
                nok_scores=data["nok_scores"],
                threshold_min=data["threshold_min"],
                threshold_max=data["threshold_max"],
                available_images=available_images,
                drift_pattern=drift_pattern,
            ))

        return analysis

    def list_nok_images(
        self,
        date_str: str,
        view_name: str,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[dict], int]:
        """Return a paginated list of NOK image metadata for a view and date.

        Each item contains the filename, absolute path, anomaly score and
        the HH:MM:SS timestamp extracted from the filename.  Results are
        sorted worst-score first (highest anomaly score at the top).

        Args:
            date_str: Date in ``YYYYMMDD`` format.
            view_name: View name to filter on.
            page: 1-based page number.
            page_size: Number of items per page.

        Returns:
            Tuple of ``(items, total)`` where ``items`` is a list of dicts
            with keys ``filename``, ``score``, ``time_str`` and ``total`` is
            the total number of matched images across all pages.
        """
        jsonl_path  = self._jsonl_path_for_date(date_str)
        records     = self._read_jsonl(jsonl_path)
        valid       = [r for r in records if r.get("view_results")]
        image_index = self._build_image_index(date_str)

        matched = self._find_nok_images(valid, view_name, image_index, date_str)
        total   = len(matched)

        start = (page - 1) * page_size
        end   = start + page_size
        page_items = matched[start:end]

        items: list[dict] = []
        for abs_path, score in page_items:
            fname = os.path.basename(abs_path)
            # Filename format: YYYYMMDDHHmmss_{view_name}.jpg
            # Extract HH:MM:SS from positions 8-14 of the timestamp prefix.
            try:
                ts_part = fname[: fname.index(f"_{view_name}.jpg")]
                time_str = f"{ts_part[8:10]}:{ts_part[10:12]}:{ts_part[12:14]}"
            except (ValueError, IndexError):
                time_str = ""
            items.append({
                "filename": fname,
                "abs_path": abs_path,
                "score":    round(score, 6),
                "time_str": time_str,
            })

        return items, total

    def list_all_nok_images_for_view(
        self,
        view_name: str,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[dict], int]:
        """Return a paginated list of NOK image metadata for a view, across
        every date found in the traceability history — no date filter.

        Used by the "View Production NOK" viewer (Step 1 of the calibration
        page) so operators can browse, and delete to free up disk space, NOK
        inference images without first having to pick a date, unlike
        :meth:`list_nok_images` (used by the date-scoped, promote-oriented
        "Recalibrate from Production" flow). Results are sorted newest-first
        (unlike :meth:`list_nok_images`'s worst-score-first order, which
        exists specifically to support hard-example mining during promote).

        Args:
            view_name: View name to filter on.
            page: 1-based page number.
            page_size: Number of items per page.

        Returns:
            Tuple of ``(items, total)`` — same item shape as
            :meth:`list_nok_images` plus a ``"date_str"`` key.
        """
        all_matched: list[tuple[str, float, str]] = []

        for date_str in self._all_traceability_dates():
            jsonl_path = self._jsonl_path_for_date(date_str)
            records    = self._read_jsonl(jsonl_path)
            valid      = [r for r in records if r.get("view_results")]

            # Cheap pre-check so a day with no NOK record at all for this view
            # never pays for building the (filesystem-listing) image index.
            has_nok_for_view = any(
                vr.get("view_name") == view_name and vr.get("classification") == "NOK"
                for r in valid for vr in r.get("view_results", [])
            )
            if not has_nok_for_view:
                continue

            image_index = self._build_image_index(date_str)
            matched = self._find_nok_images(valid, view_name, image_index, date_str)
            all_matched.extend((p, s, date_str) for p, s in matched)

        # Newest first — filenames are timestamp-prefixed.
        all_matched.sort(key=lambda x: os.path.basename(x[0]), reverse=True)
        total = len(all_matched)

        start = (page - 1) * page_size
        page_items = all_matched[start:start + page_size]

        items: list[dict] = []
        for abs_path, score, date_str in page_items:
            fname = os.path.basename(abs_path)
            try:
                ts_part = fname[: fname.index(f"_{view_name}.jpg")]
                time_str = f"{ts_part[8:10]}:{ts_part[10:12]}:{ts_part[12:14]}"
            except (ValueError, IndexError):
                time_str = ""
            items.append({
                "filename": fname,
                "abs_path": abs_path,
                "score":    round(score, 6),
                "date_str": date_str,
                "time_str": time_str,
            })

        return items, total

    def delete_nok_images(self, paths: list[str]) -> dict:
        """Permanently delete inference images from disk (disk-space cleanup).

        Every path is validated to resolve inside ``inference_images_path``
        before any filesystem operation — rejects paths outside that tree
        instead of silently following them (mirrors how
        ``CaptureReviewService.delete_images`` guards the calibration image
        tree). Irreversible — the UI only reaches this after an explicit
        operator confirmation modal.

        Args:
            paths: Absolute paths of images to delete.

        Returns:
            dict: ``{"deleted": int, "errors": list[str]}``.
        """
        deleted = 0
        errors: list[str] = []
        inference_root = os.path.realpath(self._inference_images_path)
        for path in paths:
            try:
                path_abs = os.path.realpath(path)
                if not path_abs.startswith(inference_root + os.sep):
                    errors.append(f"{path}: outside inference_images_path, skipped.")
                    continue
                if not os.path.isfile(path_abs):
                    errors.append(f"{path}: file not found.")
                    continue
                os.remove(path_abs)
                deleted += 1
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        return {"deleted": deleted, "errors": errors}

    def promote_images(
        self,
        date_str: str,
        view_names: list[str],
        test_count: int = 5,
        train_count: int = 20,
        excluded_filenames: set[str] | None = None,
    ) -> dict[str, dict[str, int]]:
        """Move NOK inference images for selected views into the training tree.

        For each view in ``view_names``:
        - Finds every inference image that was classified as NOK for that view
          on the given date.
        - Moves the ``train_count`` images with the **worst anomaly score**
          (highest score = farthest from Gaussian centroid) to
          ``{images_path}/train/OK/{view}/``. Train has priority: when few
          images are available they all go to train rather than test.
        - Moves the most recent ``test_count`` images from the **remainder**
          (after train) to ``{images_path}/test/OK/{view}/``.

        Images are moved, not copied: once promoted, they no longer clutter
        ``inference_images_path`` (avoids permanently duplicating disk usage
        for every promoted image). This is safe because the operator can
        still browse every promoted image afterwards — it just lives under
        the Step 1 "Train OK"/"Test OK" tabs instead of the original
        production folder — and can still browse every *non*-promoted NOK
        image via :meth:`list_all_nok_images_for_view`.

        Args:
            date_str: Date whose images should be promoted (``YYYYMMDD``).
            view_names: Views to promote.  Must be confirmed by the operator.
            test_count: Number of images to keep in ``test/OK``; the rest go to
                ``train/OK``.  Defaults to 5.
            train_count: Maximum number of images to copy to ``train/OK`` (most
                recent ones are preferred).  Defaults to 20.  Keeps the
                training set small to avoid unbalancing the Gaussian model.
            excluded_filenames: Optional set of bare filenames (basenames) to
                skip.  The operator can deselect individual images in the gallery
                before promoting.  Filenames not present in the matched set are
                silently ignored.

        Returns:
            dict mapping ``view_name`` →
            ``{"train": n_train_moved, "test": n_test_moved, "skipped": n_skipped}``.
        """
        excluded = excluded_filenames or set()
        jsonl_path = self._jsonl_path_for_date(date_str)
        records    = self._read_jsonl(jsonl_path)
        valid      = [r for r in records if r.get("view_results")]

        image_index = self._build_image_index(date_str)
        result: dict[str, dict[str, int]] = {}

        for vname in view_names:
            # Each item is (abs_path, score) sorted worst-score first.
            all_matched = self._find_nok_images(valid, vname, image_index, date_str)
            # Remove operator-excluded images.
            matched = [
                (p, s) for p, s in all_matched
                if os.path.basename(p) not in excluded
            ]

            # ── train/OK: worst-score train_count images (hard-example mining) ─
            # Train has priority: pick the highest anomaly-score images first so
            # the Gaussian is stretched toward the false-positive boundary.
            # When few images are available they all go to train, not to test.
            by_score    = sorted(matched, key=lambda x: x[1], reverse=True)
            train_items = by_score[:train_count]
            rest_items  = by_score[train_count:]

            # ── test/OK: most-recent test_count images from the remainder ────────
            # Pick the latest production samples from what train didn't take.
            by_time    = sorted(rest_items, key=lambda x: os.path.basename(x[0]), reverse=True)
            test_items = by_time[:test_count]

            train_paths = [p for p, _ in train_items]
            test_paths  = [p for p, _ in test_items]

            train_dest = os.path.join(self._images_path, "train", "OK", vname)
            test_dest  = os.path.join(self._images_path, "test",  "OK", vname)
            os.makedirs(train_dest, exist_ok=True)
            os.makedirs(test_dest,  exist_ok=True)

            n_train = self._move_images(train_paths, train_dest)
            n_test  = self._move_images(test_paths,  test_dest)
            n_skip  = len(matched) - n_train - n_test

            result[vname] = {"train": n_train, "test": n_test, "skipped": n_skip}
            print(f"[INFO] TraceabilityReviewService: promoted {vname}: "
                  f"train={n_train}, test={n_test}, skipped={n_skip} "
                  f"(cap: train≤{train_count}, test≤{test_count})")

        return result

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _jsonl_path_for_date(self, date_str: str) -> str:
        """Return the expected absolute path to the JSONL file for ``date_str``.

        Args:
            date_str: Date in ``YYYYMMDD`` format.

        Returns:
            str: Absolute path, which may or may not exist on disk.
        """
        month_str = date_str[:6]
        return os.path.join(
            self._traceability_path, month_str, f"{date_str}_results.jsonl"
        )

    def _all_traceability_dates(self) -> list[str]:
        """Return every ``YYYYMMDD`` date with a traceability JSONL file, sorted.

        Scans ``{traceability_path}/YYYYMM/*_results.jsonl`` — used by
        :meth:`list_all_nok_images_for_view` to scan the full history instead
        of a single operator-picked date.

        Returns:
            list[str]: Dates in ``YYYYMMDD`` format, ascending.
        """
        dates: list[str] = []
        if not os.path.isdir(self._traceability_path):
            return dates
        for month_dir in os.listdir(self._traceability_path):
            month_path = os.path.join(self._traceability_path, month_dir)
            if not os.path.isdir(month_path):
                continue
            for fname in os.listdir(month_path):
                if fname.endswith("_results.jsonl"):
                    dates.append(fname[: len("YYYYMMDD")])
        return sorted(dates)

    def _read_jsonl(self, path: str) -> list[dict]:
        """Read and parse all valid JSON lines from a JSONL file.

        Silently skips lines that are not valid JSON.

        Args:
            path: Absolute path to the ``.jsonl`` file.

        Returns:
            list[dict]: Parsed records.  Empty list if the file does not exist.
        """
        if not os.path.isfile(path):
            return []
        records = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    def _build_image_index(self, date_str: str) -> dict[tuple[str, str], str]:
        """Build a ``(part_id, view_name)`` → absolute-path mapping for a day.

        Scans ``{inference_images_path}/YYYYMM/YYYYMMDD/`` for ``date_str``, and
        also the following calendar day's folder. A cycle's JSONL record is
        filed under the day its ``date_inspected`` (cycle start) falls on, but
        the matching image is saved later, at cycle end — which can land in the
        next day's folder if the piece-wait step happened to straddle midnight.

        Args:
            date_str: Date in ``YYYYMMDD`` format.

        Returns:
            dict mapping ``(part_id, view_name)`` to the absolute image path.
        """
        index: dict[tuple[str, str], str] = {}
        for day in (date_str, self._next_day_str(date_str)):
            month_str = day[:6]
            day_dir   = os.path.join(self._inference_images_path, month_str, day)
            if not os.path.isdir(day_dir):
                continue
            for fname in os.listdir(day_dir):
                match = _IMAGE_FILENAME_RE.match(fname)
                if not match:
                    continue
                part_id, view_name = match.groups()
                index[(part_id, view_name)] = os.path.join(day_dir, fname)
        return index

    @staticmethod
    def _next_day_str(date_str: str) -> str:
        """Return the calendar day after ``date_str`` (``YYYYMMDD`` in/out)."""
        return (datetime.strptime(date_str, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

    def _find_nok_images(
        self,
        records: list[dict],
        view_name: str,
        image_index: dict[tuple[str, str], str],
        date_str: str,
    ) -> list[tuple[str, float]]:
        """Return ``(abs_path, score)`` pairs for NOK occurrences of ``view_name``.

        Each NOK record is matched to its image by the exact ``(part_id,
        view_name)`` key — the image filename embeds the ``part_id`` (see
        ``LocalStorageAdapter.save_frames``) — so there is no ambiguity from
        guessing a time window. A record with no matching image on disk is
        simply skipped.

        Pairs are sorted by score **descending** (highest/worst score first)
        to support hard-example mining in :meth:`promote_images`.

        Args:
            records: All valid records for the day.
            view_name: View name to filter on.
            image_index: ``(part_id, view_name)`` → path mapping from
                :meth:`_build_image_index`.
            date_str: Date in ``YYYYMMDD`` format (unused; kept so the method
                signature matches its callers/tests).

        Returns:
            list[tuple[str, float]]: ``(abs_path, score)`` sorted worst-first.
        """
        items: list[tuple[str, float]] = []

        for record in records:
            nok_score: float | None = None
            for vr in record.get("view_results", []):
                if vr.get("view_name") == view_name and vr.get("classification") == "NOK":
                    nok_score = float(vr.get("score", 0.0))
                    break
            if nok_score is None:
                continue

            part_id = record.get("part", {}).get("part_id")
            if not part_id:
                continue

            fpath = image_index.get((part_id, view_name))
            if fpath is None:
                continue

            items.append((fpath, nok_score))

        # Sort worst (highest score) first for hard-example mining.
        return sorted(items, key=lambda x: x[1], reverse=True)

    def _count_matching_images(
        self,
        records: list[dict],
        view_name: str,
        image_index: dict[tuple[str, str], str],
        date_str: str,
    ) -> int:
        """Count available images for NOK events of a view without copying.

        Args:
            records: All valid records for the day.
            view_name: View name to count for.
            image_index: ``(part_id, view_name)`` → path mapping.
            date_str: Date string.

        Returns:
            int: Number of matched image files.
        """
        return len(self._find_nok_images(records, view_name, image_index, date_str))

    @staticmethod
    def _move_images(src_paths: list[str], dest_dir: str) -> int:
        """Move a list of image files into ``dest_dir``.

        If a file with the same basename already exists at the destination
        (e.g. an accidental re-promote of the same date/view), the source is
        simply removed instead of overwriting it — the content is identical,
        so keeping both would just be the duplicate this method exists to
        avoid. Uses an atomic rename (same filesystem in practice, since both
        trees live under ``data/``) rather than copy+delete.

        Args:
            src_paths: Absolute source paths.
            dest_dir: Destination directory (must already exist).

        Returns:
            int: Number of files successfully moved (or de-duplicated away).
        """
        moved = 0
        for src in src_paths:
            basename = os.path.basename(src)
            dest     = os.path.join(dest_dir, basename)
            if os.path.exists(dest):
                # Already promoted earlier — drop the now-redundant source
                # copy instead of leaving a stray duplicate behind.
                try:
                    os.remove(src)
                except Exception as exc:
                    print(f"[WARN] TraceabilityReviewService: could not remove "
                          f"duplicate source '{basename}': {exc}")
                moved += 1
                continue
            try:
                shutil.move(src, dest)
                moved += 1
            except Exception as exc:
                print(f"[ERROR] TraceabilityReviewService: could not move "
                      f"'{basename}' to '{dest_dir}': {exc}")
        return moved
