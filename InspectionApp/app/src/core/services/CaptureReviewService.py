import os


class CaptureReviewService:
    """
    Pure core service for reviewing and relabeling capture-mode images
    (Train OK / Test OK / Test NOK / Discarded) collected via Samples mode
    or the (legacy) calibration capture flow.

    Operates entirely on the filesystem tree rooted at ``images_path``::

        {images_path}/train/OK/{view_name}/*.jpg
        {images_path}/test/OK/{view_name}/*.jpg
        {images_path}/test/NOK/{view_name}/*.jpg
        {images_path}/discarded/{view_name}/*.jpg

    No hardware dependency, so this stays in ``core/`` per the architecture
    invariants (services depend only on interfaces + core).

    Discarding is reversible: moving an image to ``discarded`` and moving it
    back are both plain ``relabel_images()`` calls to a different
    ``target_set``. ``delete_images()`` is the one destructive operation in
    this service — it permanently removes files from disk and is only meant
    to be reached after an explicit operator confirmation in the UI.
    """

    _SET_SUBDIRS: dict[str, str] = {
        "train_ok":  os.path.join("train", "OK"),
        "test_ok":   os.path.join("test",  "OK"),
        "test_nok":  os.path.join("test",  "NOK"),
        "discarded": "discarded",
    }

    def __init__(self, images_path: str):
        """
        Args:
            images_path (str): Root directory for the image tree (same value
                used by ``SampleCaptureService``/``CalibrationService``).
        """
        self._images_path = os.path.abspath(images_path)

    def _set_dir(self, target_set: str, view_name: str) -> str:
        """Resolve the absolute directory for a (target_set, view_name) pair."""
        if target_set not in self._SET_SUBDIRS:
            raise ValueError(f"Unknown set '{target_set}'.")
        return os.path.join(self._images_path, self._SET_SUBDIRS[target_set], view_name)

    def list_images(
        self, target_set: str, view_name: str, page: int = 1, page_size: int = 50
    ) -> tuple[list[dict], int]:
        """
        Return a paginated, newest-first list of images for one (set, view).

        Args:
            target_set (str): One of ``'train_ok'``, ``'test_ok'``,
                ``'test_nok'``, ``'discarded'``.
            view_name (str): View name (matches the per-view subfolder).
            page (int): 1-based page number.
            page_size (int): Items per page.

        Returns:
            tuple[list[dict], int]: ``(items, total)`` — items are
                ``{"filename": str, "abs_path": str}``, newest first
                (filenames are timestamp-prefixed, so a reverse sort by name
                is a reverse-chronological sort).
        """
        directory = self._set_dir(target_set, view_name)
        if not os.path.isdir(directory):
            return [], 0

        filenames = sorted(
            (f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".jpeg", ".png"))),
            reverse=True,
        )
        total = len(filenames)
        start = (page - 1) * page_size
        page_files = filenames[start:start + page_size]
        items = [
            {"filename": fn, "abs_path": os.path.join(directory, fn)}
            for fn in page_files
        ]
        return items, total

    def relabel_images(self, paths: list[str], target_set: str) -> dict:
        """
        Move each given absolute image path into ``target_set``, keeping the
        same per-view subfolder (the view name is derived from the source
        path's parent directory name, so this works regardless of which set
        the image is currently in). Atomic move via ``os.replace``.

        Every path is validated to resolve inside ``images_path`` before any
        filesystem operation — rejects paths outside the current sequence's
        image tree instead of silently following them.

        Args:
            paths (list[str]): Absolute paths of images to move.
            target_set (str): Destination set — one of ``'train_ok'``,
                ``'test_ok'``, ``'test_nok'``, ``'discarded'``.

        Returns:
            dict: ``{"moved": int, "errors": list[str]}``.
        """
        if target_set not in self._SET_SUBDIRS:
            raise ValueError(f"Unknown set '{target_set}'.")

        moved = 0
        errors: list[str] = []
        for src in paths:
            try:
                src_abs = os.path.realpath(src)
                if not src_abs.startswith(self._images_path + os.sep):
                    errors.append(f"{src}: outside the current sequence's images_path, skipped.")
                    continue
                if not os.path.isfile(src_abs):
                    errors.append(f"{src}: file not found.")
                    continue

                view_name = os.path.basename(os.path.dirname(src_abs))
                dest_dir  = self._set_dir(target_set, view_name)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(src_abs))
                os.replace(src_abs, dest_path)
                moved += 1
            except Exception as exc:
                errors.append(f"{src}: {exc}")

        return {"moved": moved, "errors": errors}

    def delete_images(self, paths: list[str]) -> dict:
        """
        Permanently delete each given absolute image path from disk.

        Every path is validated to resolve inside ``images_path`` before any
        filesystem operation — rejects paths outside the current sequence's
        image tree instead of silently following them. This is a destructive,
        irreversible operation: callers must obtain explicit operator
        confirmation before invoking it (the UI does this via a confirmation
        modal — see ``/api/review/delete``).

        Args:
            paths (list[str]): Absolute paths of images to delete.

        Returns:
            dict: ``{"deleted": int, "errors": list[str]}``.
        """
        deleted = 0
        errors: list[str] = []
        for path in paths:
            try:
                path_abs = os.path.realpath(path)
                if not path_abs.startswith(self._images_path + os.sep):
                    errors.append(f"{path}: outside the current sequence's images_path, skipped.")
                    continue
                if not os.path.isfile(path_abs):
                    errors.append(f"{path}: file not found.")
                    continue

                os.remove(path_abs)
                deleted += 1
            except Exception as exc:
                errors.append(f"{path}: {exc}")

        return {"deleted": deleted, "errors": errors}
