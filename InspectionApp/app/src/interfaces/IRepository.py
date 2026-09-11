from abc import ABC, abstractmethod

from app.src.core.models.Part import Part


class IRepository(ABC):
    """Abstract interface for repository operations."""

    @abstractmethod
    def save_inspection_result(self, part: Part) -> bool:
        """
        Save the inspection result for a given part.

        Args:
            part (Part): The completed Part object with inspection results.

        Returns:
            bool: True if saved successfully, False otherwise.
        """
        pass

    @abstractmethod
    def get_inspection_result(self, part_id: str) -> Part | None:
        """
        Retrieve a previously saved Part by its ID.

        Args:
            part_id (str): Unique identifier for the part.

        Returns:
            Part | None: The Part object if found, None otherwise.
        """
        pass

    @abstractmethod
    def save_frames(self, part_id: str, captured_frames: dict) -> bool:
        """
        Save the captured raw frames alongside their inspection result.

        Args:
            part_id (str): Unique identifier for the part.
            captured_frames (dict[str, np.ndarray]): Frames keyed by view_name.

        Returns:
            bool: True if all frames were saved successfully, False otherwise.
        """
        pass

