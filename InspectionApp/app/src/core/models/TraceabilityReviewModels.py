from dataclasses import dataclass, field


@dataclass
class ViewReviewStats:
    """Per-view statistics computed from a traceability JSONL file.

    Attributes:
        view_name: Canonical view name (e.g. ``section_2_B``).
        total_parts: Total parts inspected that included this view.
        nok_count: How many of those parts classified this view as NOK.
        nok_rate: ``nok_count / total_parts``.
        all_scores: Anomaly scores for every part (OK and NOK alike).
        nok_scores: Anomaly scores only for NOK-classified parts.
        threshold_min: Lower acceptance threshold stored in the last record.
        threshold_max: Upper acceptance threshold stored in the last record.
        available_images: How many NOK-classified image files were found on disk.
        drift_pattern: ``"global"`` when this view's NOK events are correlated
            with a majority of other views failing in the same cycle (systemic
            calibration drift), ``"isolated"`` when this view fails alone, or
            ``"none"`` when the NOK rate is zero.
    """

    view_name: str
    total_parts: int
    nok_count: int
    nok_rate: float
    all_scores: list[float] = field(default_factory=list)
    nok_scores: list[float] = field(default_factory=list)
    threshold_min: float = 0.0
    threshold_max: float = 0.0
    available_images: int = 0
    drift_pattern: str = "none"  # "global" | "isolated" | "none"


@dataclass
class ReviewAnalysis:
    """Result of analyzing a single day's traceability file.

    Attributes:
        date_str: Date analyzed in ``YYYYMMDD`` format.
        total_parts: Total parts with complete ``view_results`` (errors excluded).
        total_nok_parts: Parts whose overall result was NOK (at least one view
            classified as NOK).  Used for the global NOK rate displayed in the
            review modal header.
        view_stats: Per-view statistics, one entry per view name found.
        global_drift_detected: ``True`` if at least one cycle was found where
            ≥ 5 views failed simultaneously — strong signal of systemic drift.
        jsonl_path: Absolute path of the JSONL file that was analyzed.
    """

    date_str: str
    total_parts: int
    total_nok_parts: int = 0
    view_stats: list[ViewReviewStats] = field(default_factory=list)
    global_drift_detected: bool = False
    jsonl_path: str = ""
