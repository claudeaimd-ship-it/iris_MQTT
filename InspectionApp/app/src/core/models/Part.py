from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

@dataclass
class InspectionResult:
    """
    Represents the result of a single AI view inspection.

    Note:
        Defined in Part.py because it is only used by Part.

    Attributes:
        view (str): Name of the inspected view (e.g. 'front_A', 'back_B').
        is_ok (bool): True if the inspection passed, False otherwise.
        score (float): Anomaly score produced by the model.
        threshold_used (tuple[float, float]): Score range (min, max) used as acceptance threshold.
        error_map (np.ndarray | None): Pixel-level error map from the model, if available.
    """
    view: str
    is_ok: bool
    score: float
    threshold_used: tuple[float, float]
    error_map: np.ndarray | None = None


@dataclass
class TriggerEvent:
    """
    Records a single GPIO interaction during the inspection sequence.

    Used for quality traceability: allows auditing at which step the sequence
    completed, was interrupted, or produced unexpected results.

    Attributes:
        step_number (int): The sequence step in which the event occurred.
        direction (str): 'INPUT' for signals received, 'OUTPUT' for signals sent.
        pin (int): GPIO pin number involved.
        action (str): The GPIO action performed ('turn_on', 'turn_off',
            'send_output', 'wait_for_input').
        result (str): Outcome of the action: 'SENT' for outputs,
            'OK' or 'TIMEOUT' for inputs.
    """
    step_number: int
    direction: str
    pin: int
    action: str
    result: str


class Part:
    """
    Represents the part being inspected.

    Holds the captured view images, inspection results, and derived status
    for a single part throughout one inspection cycle.

    Attributes:
        part_id (str): Unique identifier for this specific physical part instance.
        model_id (str): Identifier of the part model (type), matching ``part_model``
            from the sequence JSON. Links the part to the ``part_models`` DB table.
        date_inspected (datetime): Timestamp of when the inspection started.
        time_inspected (float | None): Total inspection duration in seconds. None until completed.
        inspection_results (list[InspectionResult]): Results from each inspected view.
        triggers (list[TriggerEvent]): GPIO events recorded during the sequence for traceability.
        piece_detected (bool | None): Whether a piece was detected. None until evaluated.
        dry_run (bool): True when the cycle was executed in dry-run mode (NOK steps skipped).
    """

    def __init__(self, part_id: str, model_id: str):
        """
        Args:
            part_id (str): Unique identifier for this specific physical part instance.
            model_id (str): Identifier of the part model (type), e.g. ``'nissan_shroud'``.
        """
        self.part_id = part_id
        self.model_id = model_id
        self.date_inspected = datetime.now()
        self.time_inspected: float | None = None
        self.inspection_results: list[InspectionResult] = []
        self.triggers: list[TriggerEvent] = []
        self.piece_detected: bool | None = None
        self.dry_run: bool = False
        self.forced_scrap: bool = False
        self.failed_channel: str | None = None  # Camera channel that caused a system_error_paused abort, if any.
        self.failed_channel_error: str | None = None

    @property
    def overall_status(self) -> bool:
        """
        Determine the overall status of the part based on inspection results.

        Returns:
            bool: True if all inspections passed, False otherwise.
        """
        if not self.inspection_results:  # No results means the inspection was not completed.
            return False
        return all(result.is_ok for result in self.inspection_results)