from dataclasses import dataclass, field


@dataclass
class BlockSweepResult:
    """
    Results for a single PaDiM backbone block evaluated during a sweep.

    Attributes:
        block (int): MobileNetV2 block number (3–17).
        auc (float): Area under the ROC curve (0.0–1.0). 1.0 = perfect separation.
        sep_ratio (float): Separation ratio: ``min(NOK_scores) / max(OK_scores)``.
            Values > 1.25 indicate reliable separation.
        min_nok (float): Minimum anomaly score across all test NOK images.
        max_ok (float): Maximum anomaly score across all test OK images.
        min_ok (float): Minimum anomaly score across all test OK images.
        mean_ok (float): Mean anomaly score across all test OK images.
        std_ok (float): Standard deviation of anomaly scores across all test
            OK images. Lower values mean a tighter, more consistent OK
            distribution for this block.
        cv_ok (float): Coefficient of variation of the OK scores
            (``std_ok / mean_ok``). Used to compare how stable the OK
            distribution is between blocks, independently of ``sep_ratio``.
        n_train_ok (int): Number of training OK images used to fit the Gaussian.
        n_test_ok (int): Number of test OK images scored.
        n_test_nok (int): Number of test NOK images scored.
    """

    block:      int
    auc:        float
    sep_ratio:  float
    min_nok:    float
    max_ok:     float
    min_ok:     float
    mean_ok:    float
    std_ok:     float
    cv_ok:      float
    n_train_ok: int
    n_test_ok:  int
    n_test_nok: int


@dataclass
class SweepResult:
    """
    Aggregated results from a full block sweep across all configured views.

    Attributes:
        view_name (str): The view this sweep was run for
            (e.g. ``"front_view_section_1_A"``).
        block_results (list[BlockSweepResult]): One entry per block evaluated,
            sorted by block number.
        best_block (int): Selected block. Among the candidates whose
            ``sep_ratio`` passes ``min_sep_gate``, this is the one with the
            lowest ``cv_ok`` (most consistent OK distribution). If no block
            passes the gate, falls back to the block with the highest
            ``sep_ratio`` (previous behavior).
        best_sep_ratio (float): Separation ratio of the selected block.
        best_auc (float): AUC of the selected block.
        best_cv_ok (float): Coefficient of variation of the OK scores for the
            selected block.
    """

    view_name:      str
    block_results:  list[BlockSweepResult] = field(default_factory=list)
    best_block:     int   = 0
    best_sep_ratio: float = 0.0
    best_auc:       float = 0.0
    best_cv_ok:     float = 0.0


@dataclass
class CalibrationResult:
    """
    Result of a final PaDiM calibration run for a single view.

    Attributes:
        view_name (str): View name (e.g. ``"front_view_section_1_A"``).
        block (int): MobileNetV2 block used.
        sep_ratio (float): Separation ratio achieved on the test set.
        auc (float): AUC on the test set.
        threshold_min (float): Lower acceptance threshold written to
            ``thresholds.json``.
        threshold_max (float): Upper acceptance threshold written to
            ``thresholds.json``.
        params_path (str): Absolute path to the written
            ``padim_{view_name}_params.npz`` file.
        eval_config_path (str): Absolute path to the written
            ``eval_config.json`` file.
        thresholds_path (str): Absolute path to the written
            ``thresholds.json`` file.
        inference_type (str): ``"standard"`` (default) or
            ``"presence_detection_absence_calibrated"`` — mirrors the
            ``inference_type`` field of the view's sequence JSON entry, kept
            here purely for audit trail in ``eval_config.json``/
            ``calibration_eval.json``; does not affect the fit itself.
    """

    view_name:        str
    block:            int
    sep_ratio:        float
    auc:              float
    threshold_min:    float
    threshold_max:    float
    params_path:      str
    eval_config_path: str
    thresholds_path:  str
    inference_type:   str = "standard"
