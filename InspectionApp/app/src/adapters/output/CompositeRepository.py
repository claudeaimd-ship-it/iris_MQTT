from app.src.interfaces.IRepository import IRepository
from app.src.core.models.Part import Part


class CompositeRepository(IRepository):
    """
    Fans out IRepository calls to multiple concrete repositories.

    Lets AppFactory wire in any number of persistence/publish destinations
    (disk, MQTT, later a DB adapter, etc.) while InspectionService keeps
    depending on the single IRepository interface, unchanged.

    A failure in one repository never stops the others from being called —
    e.g. a broker outage must never block writing the local JSONL log, and
    vice versa.

    Attributes:
        _repositories (list[IRepository]): Repositories called in order on
            every method, each independently.
    """

    def __init__(self, repositories: list[IRepository]):
        """
        Args:
            repositories (list[IRepository]): One or more concrete repository
                adapters, in the order they should be called.

        Raises:
            ValueError: If the list is empty.
        """
        if not repositories:
            raise ValueError("CompositeRepository requires at least one repository.")
        self._repositories = repositories

    # =========================================================================
    # IRepository interface
    # =========================================================================

    def save_inspection_result(self, part: Part) -> bool:
        """
        Call save_inspection_result on every wrapped repository.

        Args:
            part (Part): Completed Part to persist/publish.

        Returns:
            bool: True only if every repository succeeded.
        """
        all_ok = True
        for repo in self._repositories:
            try:
                if not repo.save_inspection_result(part):
                    all_ok = False
            except Exception as e:
                print(f"[ERROR] CompositeRepository: {repo.__class__.__name__}.save_inspection_result raised: {e}")
                all_ok = False
        return all_ok

    def get_inspection_result(self, part_id: str) -> Part | None:
        """
        Return the first non-None result from the wrapped repositories, in order.

        Args:
            part_id (str): Unique identifier for the part.

        Returns:
            Part | None: The first successful lookup, or None if none found it.
        """
        for repo in self._repositories:
            try:
                result = repo.get_inspection_result(part_id)
                if result is not None:
                    return result
            except Exception as e:
                print(f"[ERROR] CompositeRepository: {repo.__class__.__name__}.get_inspection_result raised: {e}")
        return None

    def save_frames(self, part_id: str, captured_frames: dict) -> bool:
        """
        Call save_frames on every wrapped repository.

        Args:
            part_id (str): Unique identifier for the part.
            captured_frames (dict): Frames keyed by view_name.

        Returns:
            bool: True only if every repository succeeded.
        """
        all_ok = True
        for repo in self._repositories:
            try:
                if not repo.save_frames(part_id, captured_frames):
                    all_ok = False
            except Exception as e:
                print(f"[ERROR] CompositeRepository: {repo.__class__.__name__}.save_frames raised: {e}")
                all_ok = False
        return all_ok

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """
        Call close() on every wrapped repository that implements it.

        Safe no-op for repositories without a close() method (e.g. LocalStorageAdapter).

        Returns:
            None
        """
        for repo in self._repositories:
            closer = getattr(repo, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception as e:
                    print(f"[WARN] CompositeRepository: error closing {repo.__class__.__name__}: {e}")