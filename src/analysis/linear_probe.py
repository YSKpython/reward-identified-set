"""Linear probe analysis for feature-space vulnerability audit pipeline.

This module implements a pure scikit-learn / numpy analysis for testing
the three-level pattern hypothesis (Experiment E20) from the NeurIPS 2027
submission on feature-space vulnerability in RLHF reward models.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class ProbeResult:
    """Results from a linear probe experiment.

    Attributes:
        accuracy_mean: Mean accuracy across cross-validation folds.
        accuracy_std: Standard deviation of fold accuracies.
        fold_accuracies: Per-fold accuracy values as a numpy array.
        n_folds: Number of cross-validation folds used.
        n_samples: Total number of samples in the dataset.
        feature_type: Descriptor of the input feature type
            (e.g., "pooled_hidden_state", "tfidf", "random_projection",
            "permuted_label").
    """

    accuracy_mean: float
    accuracy_std: float
    fold_accuracies: np.ndarray
    n_folds: int
    n_samples: int
    feature_type: str


@dataclass(frozen=True)
class ThreeLevelSummary:
    """Summary of the three-level pattern analysis.

    Attributes:
        pooled_result: Probe result for pooled hidden-state features.
        tfidf_result: Probe result for TF-IDF text features.
        random_projection_result: Probe result for random projection baseline.
        permuted_label_result: Probe result for permuted label control.
        pooled_vs_chance: Difference between pooled accuracy and chance (0.5).
        tfidf_vs_pooled: Difference between TF-IDF and pooled accuracy.
        three_level_pattern_confirmed: True if pooled accuracy is within
            2 standard deviations of 0.5 AND TF-IDF accuracy exceeds pooled
            accuracy by at least 0.2.
    """

    pooled_result: ProbeResult
    tfidf_result: ProbeResult
    random_projection_result: ProbeResult
    permuted_label_result: ProbeResult
    pooled_vs_chance: float
    tfidf_vs_pooled: float
    three_level_pattern_confirmed: bool


def _run_cv_pipeline(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int,
    C: float,  # noqa: N803
    random_state: int,
    feature_type: str,
) -> ProbeResult:
    """Internal helper to run stratified k-fold CV with L2 logistic regression.

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        labels: Label vector of shape (n_samples,).
        n_folds: Number of cross-validation folds.
        C: Inverse regularization strength for logistic regression.
        random_state: Random seed for reproducibility.
        feature_type: Descriptor string for the feature type.

    Returns:
        ProbeResult containing per-fold accuracies and summary statistics.
    """
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=random_state,
    )

    fold_accuracies_list: list[float] = []

    for train_idx, test_idx in skf.split(features, labels):
        X_train = features[train_idx]  # noqa: N806
        X_test = features[test_idx]  # noqa: N806
        y_train = labels[train_idx]
        y_test = labels[test_idx]

        clf = LogisticRegression(
            penalty="l2",
            C=C,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        fold_accuracies_list.append(acc)

    fold_accuracies = np.array(fold_accuracies_list, dtype=np.float64)
    accuracy_mean = float(np.mean(fold_accuracies))
    accuracy_std = float(np.std(fold_accuracies, ddof=0))

    return ProbeResult(
        accuracy_mean=accuracy_mean,
        accuracy_std=accuracy_std,
        fold_accuracies=fold_accuracies,
        n_folds=n_folds,
        n_samples=len(labels),
        feature_type=feature_type,
    )


def run_linear_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,  # noqa: N803
    random_state: int = 42,
    feature_type: str = "pooled_hidden_state",
) -> ProbeResult:
    """Train an L2-regularized logistic regression with stratified k-fold CV.

    This function performs linear probing on pre-extracted feature matrices
    to assess how well component identity can be decoded from the features.

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        labels: Label vector of shape (n_samples,) with integer class labels.
        n_folds: Number of cross-validation folds. Must be >= 2.
        C: Inverse regularization strength for logistic regression. Must be > 0.
        random_state: Random seed for reproducibility of both CV splits and
            logistic regression solver.
        feature_type: Descriptor string for the feature type
            (default: "pooled_hidden_state").

    Returns:
        ProbeResult containing per-fold accuracies and summary statistics.

    Raises:
        ValueError: If features is not 2-D.
        ValueError: If labels is not 1-D or length mismatches features.
        ValueError: If fewer than n_folds samples per class.
        ValueError: If n_folds < 2.
        ValueError: If C <= 0.
    """
    # Validate features dimensionality
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got {features.ndim}-D array")

    # Validate labels dimensionality
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got {labels.ndim}-D array")

    # Validate length match
    if len(features) != len(labels):
        raise ValueError(
            f"features and labels length mismatch: "
            f"features has {len(features)} samples, labels has {len(labels)}"
        )

    # Validate n_folds
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    # Validate C
    if C <= 0:
        raise ValueError(f"C must be > 0, got {C}")

    # Check minimum samples per class
    unique_labels, counts = np.unique(labels, return_counts=True)
    min_samples_per_class = int(np.min(counts))
    if min_samples_per_class < n_folds:
        raise ValueError(
            f"Fewer than n_folds ({n_folds}) samples per class: "
            f"minimum class count is {min_samples_per_class}"
        )

    return _run_cv_pipeline(
        features=features,
        labels=labels,
        n_folds=n_folds,
        C=C,
        random_state=random_state,
        feature_type=feature_type,
    )


def run_tfidf_probe(
    texts: Sequence[str],
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,  # noqa: N803
    random_state: int = 42,
    max_features: int = 10000,
) -> ProbeResult:
    """Build TF-IDF features from raw text and run L2 logistic regression CV.

    This function creates TF-IDF features from raw text sequences and then
    performs linear probing to assess how well component identity can be
    decoded from textual information.

    Args:
        texts: Sequence of raw text strings.
        labels: Label vector of shape (n_samples,) with integer class labels.
        n_folds: Number of cross-validation folds. Must be >= 2.
        C: Inverse regularization strength for logistic regression. Must be > 0.
        random_state: Random seed for reproducibility.
        max_features: Maximum number of TF-IDF features to extract.

    Returns:
        ProbeResult containing per-fold accuracies and summary statistics
            with feature_type="tfidf".

    Raises:
        ValueError: If texts is empty.
        ValueError: If len(texts) != len(labels).
    """
    if len(texts) == 0:
        raise ValueError("texts cannot be empty")

    if len(texts) != len(labels):
        raise ValueError(
            f"length mismatch: texts has {len(texts)} items, "
            f"labels has {len(labels)} items"
        )

    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_features=max_features,
    )
    features = vectorizer.fit_transform(list(texts)).toarray()

    # Convert to float64 for numerical stability
    features = features.astype(np.float64)

    return _run_cv_pipeline(
        features=features,
        labels=labels,
        n_folds=n_folds,
        C=C,
        random_state=random_state,
        feature_type="tfidf",
    )


def run_random_projection_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,  # noqa: N803
    random_state: int = 42,
    n_components: int = 1024,
) -> ProbeResult:
    """Project features onto a Gaussian random matrix and run L2 logistic regression CV.

    This function serves as a null control by projecting features onto a
    random subspace without any learning. The projection uses a Gaussian
    random matrix scaled by 1/sqrt(n_components).

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        labels: Label vector of shape (n_samples,) with integer class labels.
        n_folds: Number of cross-validation folds. Must be >= 2.
        C: Inverse regularization strength for logistic regression. Must be > 0.
        random_state: Random seed for reproducibility of both the random
            projection and the logistic regression CV.
        n_components: Number of components in the random projection. Must be > 0.

    Returns:
        ProbeResult containing per-fold accuracies and summary statistics
            with feature_type="random_projection".

    Raises:
        ValueError: If n_components <= 0.
    """
    if n_components <= 0:
        raise ValueError(f"n_components must be > 0, got {n_components}")

    rng = np.random.RandomState(random_state)
    R = rng.normal(0, 1, size=(features.shape[1], n_components)) / np.sqrt(
        n_components
    )  # noqa: N806
    projected = features @ R

    return _run_cv_pipeline(
        features=projected,
        labels=labels,
        n_folds=n_folds,
        C=C,
        random_state=random_state,
        feature_type="random_projection",
    )


def run_permuted_label_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,  # noqa: N803
    random_state: int = 42,
) -> ProbeResult:
    """Permute labels and run L2 logistic regression CV as a chance-level control.

    This function provides a baseline by randomly shuffling the labels,
    which should yield chance-level performance regardless of feature quality.

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        labels: Label vector of shape (n_samples,) with integer class labels.
        n_folds: Number of cross-validation folds. Must be >= 2.
        C: Inverse regularization strength for logistic regression. Must be > 0.
        random_state: Random seed for reproducibility of both the label
            permutation and the logistic regression CV.

    Returns:
        ProbeResult containing per-fold accuracies and summary statistics
            with feature_type="permuted_label".
    """
    rng = np.random.RandomState(random_state)
    permuted_labels = rng.permutation(labels)

    return _run_cv_pipeline(
        features=features,
        labels=permuted_labels,
        n_folds=n_folds,
        C=C,
        random_state=random_state,
        feature_type="permuted_label",
    )


def compute_three_level_summary(
    pooled_features: np.ndarray,
    texts: Sequence[str],
    labels: np.ndarray,
    n_folds: int = 5,
    C: float = 1.0,  # noqa: N803
    random_state: int = 42,
) -> ThreeLevelSummary:
    """Orchestrate all four probe variants and return a ThreeLevelSummary.

    This function runs the complete three-level pattern analysis by computing:
    1. Pooled hidden-state probe (tests Level 3 representation separability)
    2. TF-IDF probe (tests if text contains component information)
    3. Random projection probe (null control)
    4. Permuted label probe (chance-level control)

    The three_level_pattern_confirmed field is True if and only if:
    1. abs(pooled_result.accuracy_mean - 0.5) <= 2 * pooled_result.accuracy_std
       (pooled accuracy is statistically indistinguishable from chance), AND
    2. tfidf_result.accuracy_mean - pooled_result.accuracy_mean >= 0.2
       (TF-IDF substantially outperforms pooled hidden-state).

    Args:
        pooled_features: Feature matrix of shape (n_samples, n_features) from
            pooled hidden-states.
        texts: Sequence of raw text strings corresponding to each sample.
        labels: Label vector of shape (n_samples,) with integer class labels.
        n_folds: Number of cross-validation folds. Must be >= 2.
        C: Inverse regularization strength for logistic regression. Must be > 0.
        random_state: Random seed for reproducibility across all probes.

    Returns:
        ThreeLevelSummary containing all probe results and pattern confirmation.
    """
    pooled_result = run_linear_probe(
        features=pooled_features,
        labels=labels,
        n_folds=n_folds,
        C=C,
        random_state=random_state,
        feature_type="pooled_hidden_state",
    )

    tfidf_result = run_tfidf_probe(
        texts=texts,
        labels=labels,
        n_folds=n_folds,
        C=C,
        random_state=random_state,
    )

    random_projection_result = run_random_projection_probe(
        features=pooled_features,
        labels=labels,
        n_folds=n_folds,
        C=C,
        random_state=random_state,
    )

    permuted_label_result = run_permuted_label_probe(
        features=pooled_features,
        labels=labels,
        n_folds=n_folds,
        C=C,
        random_state=random_state,
    )

    pooled_vs_chance = pooled_result.accuracy_mean - 0.5
    tfidf_vs_pooled = tfidf_result.accuracy_mean - pooled_result.accuracy_mean

    # Check three-level pattern conditions
    condition_1 = abs(pooled_result.accuracy_mean - 0.5) <= (
        2 * pooled_result.accuracy_std
    )
    condition_2 = tfidf_vs_pooled >= 0.2
    three_level_pattern_confirmed = condition_1 and condition_2

    return ThreeLevelSummary(
        pooled_result=pooled_result,
        tfidf_result=tfidf_result,
        random_projection_result=random_projection_result,
        permuted_label_result=permuted_label_result,
        pooled_vs_chance=pooled_vs_chance,
        tfidf_vs_pooled=tfidf_vs_pooled,
        three_level_pattern_confirmed=three_level_pattern_confirmed,
    )
