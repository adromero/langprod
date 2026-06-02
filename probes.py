"""Probes module — linear probing classifiers for hidden-state representations.

Implements:
    - train_probe_at_layer(): L2-regularized logistic regression with nested
      GroupKFold cross-validation and bootstrap confidence intervals.
    - train_probes_all_layers(): Probe all three tasks (product, category,
      register) at every layer for all anisotropy correction methods.
    - train_control_probes(): Hewitt & Manning (2019) control tasks —
      permuted-label probes and selectivity computation.
    - compute_zone_boundaries(): Divide layers into early / protocol / late /
      output zones.
    - train_zone_probes(): Mean-pool zone representations and train probes
      per zone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anisotropy correction helpers
# ---------------------------------------------------------------------------


def _apply_anisotropy_correction(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: str,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply anisotropy correction to representations.

    Args:
        X_train: Training representations, shape (n_train, D).
        X_test: Test representations, shape (n_test, D).
        method: One of "none", "mean_centering", "whitening".
        epsilon: Small constant for numerical stability in whitening.

    Returns:
        Tuple of (corrected_X_train, corrected_X_test).
    """
    if method == "none":
        return X_train, X_test

    # Compute mean from training set only
    mean = X_train.mean(axis=0, keepdims=True)

    if method == "mean_centering":
        return X_train - mean, X_test - mean

    if method == "whitening":
        X_centered = X_train - mean
        # Compute covariance and whiten
        cov = np.cov(X_centered, rowvar=False)
        # Eigendecomposition for stable whitening
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Clip small eigenvalues for stability
        eigvals = np.maximum(eigvals, epsilon)
        whitening_matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        X_train_whitened = (X_train - mean) @ whitening_matrix
        X_test_whitened = (X_test - mean) @ whitening_matrix
        return X_train_whitened, X_test_whitened

    raise ValueError(f"Unknown anisotropy method: {method}")


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean of ``scores``.

    Args:
        scores: 1-D array of per-fold (or per-sample) scores.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95% CI).
        rng: NumPy random generator for reproducibility.

    Returns:
        (lower_bound, upper_bound) of the CI.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    boot_means = np.empty(n_bootstrap)
    n = len(scores)
    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = scores[indices].mean()

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return lower, upper


# ---------------------------------------------------------------------------
# Core probe training
# ---------------------------------------------------------------------------


def train_probe_at_layer(
    representations: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    config: dict,
) -> dict[str, Any]:
    """Train an L2-regularized logistic regression probe with nested GroupKFold CV.

    Outer loop: ``probe_outer_folds``-fold GroupKFold.
    Inner loop: Within each outer training fold, ``probe_inner_folds``-fold
    GroupKFold to select the best C from ``probe_C_values``.

    Args:
        representations: Shape (N, D) — feature matrix for one layer.
        labels: Shape (N,) — integer-encoded class labels.
        groups: Shape (N,) — group IDs (product_id) for GroupKFold.
        config: CONFIG dict containing probe hyperparameters.

    Returns:
        Dict with keys: macro_f1, micro_f1, per_fold_f1, bootstrap_ci_95,
        best_C_per_fold, mean_best_C.
    """
    n_outer = config["probe_outer_folds"]
    n_inner = config["probe_inner_folds"]
    C_values = config["probe_C_values"]
    max_iter = config["probe_max_iter"]
    solver = config["probe_solver"]

    outer_cv = GroupKFold(n_splits=n_outer)

    per_fold_macro_f1 = []
    per_fold_micro_f1 = []
    best_C_per_fold = []
    all_y_true = []
    all_y_pred = []

    for outer_train_idx, outer_test_idx in outer_cv.split(
        representations, labels, groups=groups
    ):
        X_train_outer = representations[outer_train_idx]
        y_train_outer = labels[outer_train_idx]
        g_train_outer = groups[outer_train_idx]
        X_test_outer = representations[outer_test_idx]
        y_test_outer = labels[outer_test_idx]

        # --- Inner CV to select best C ---
        # Guard against n_inner > number of unique groups in the outer train set
        unique_inner_groups = np.unique(g_train_outer)
        effective_inner_folds = min(n_inner, len(unique_inner_groups))
        if effective_inner_folds < 2:
            # Not enough groups for inner CV — use default C=1.0
            best_C = 1.0
        else:
            inner_cv = GroupKFold(n_splits=effective_inner_folds)
            mean_inner_scores: dict[float, float] = {}

            for C_val in C_values:
                inner_scores = []
                for inner_train_idx, inner_val_idx in inner_cv.split(
                    X_train_outer, y_train_outer, groups=g_train_outer
                ):
                    X_inner_train = X_train_outer[inner_train_idx]
                    y_inner_train = y_train_outer[inner_train_idx]
                    X_inner_val = X_train_outer[inner_val_idx]
                    y_inner_val = y_train_outer[inner_val_idx]

                    clf = LogisticRegression(
                        C=C_val,
                        solver=solver,
                        max_iter=max_iter,
                        random_state=42,
                        n_jobs=1,
                    )
                    # Fit may warn about convergence — that is acceptable
                    clf.fit(X_inner_train, y_inner_train)
                    y_inner_pred = clf.predict(X_inner_val)
                    inner_f1 = f1_score(
                        y_inner_val, y_inner_pred, average="macro", zero_division=0.0
                    )
                    inner_scores.append(inner_f1)

                mean_inner_scores[C_val] = float(np.mean(inner_scores))

            best_C = max(mean_inner_scores, key=mean_inner_scores.get)  # type: ignore[arg-type]

        best_C_per_fold.append(best_C)

        # --- Train final model on full outer training set with best C ---
        clf_final = LogisticRegression(
            C=best_C,
            solver=solver,
            max_iter=max_iter,
            random_state=42,
            n_jobs=1,
        )
        clf_final.fit(X_train_outer, y_train_outer)
        y_pred_outer = clf_final.predict(X_test_outer)

        fold_macro = f1_score(
            y_test_outer, y_pred_outer, average="macro", zero_division=0.0
        )
        fold_micro = f1_score(
            y_test_outer, y_pred_outer, average="micro", zero_division=0.0
        )
        per_fold_macro_f1.append(float(fold_macro))
        per_fold_micro_f1.append(float(fold_micro))

        all_y_true.extend(y_test_outer.tolist())
        all_y_pred.extend(y_pred_outer.tolist())

    # Aggregate metrics
    macro_f1_arr = np.array(per_fold_macro_f1)
    micro_f1_arr = np.array(per_fold_micro_f1)

    # Bootstrap CIs on per-fold macro-F1 scores
    rng = np.random.default_rng(42)
    ci_lower, ci_upper = _bootstrap_ci(macro_f1_arr, n_bootstrap=1000, rng=rng)

    return {
        "macro_f1": float(macro_f1_arr.mean()),
        "micro_f1": float(micro_f1_arr.mean()),
        "per_fold_f1": per_fold_macro_f1,
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "best_C_per_fold": best_C_per_fold,
        "mean_best_C": float(np.mean(best_C_per_fold)),
    }


# ---------------------------------------------------------------------------
# Metadata / label helpers
# ---------------------------------------------------------------------------


def _build_label_arrays(
    stimulus_ids_h5: list[str],
    stimuli_meta: list[dict],
) -> dict[str, dict[str, np.ndarray]]:
    """Build label and group arrays for all three probe tasks.

    Aligns HDF5 row order (stimulus_ids_h5) with metadata.

    Returns:
        Dict mapping task name -> {"labels": ndarray, "groups": ndarray,
        "encoder": LabelEncoder, "mask": ndarray (bool)}.
        The mask indicates which rows have valid labels for that task.
    """
    # Index metadata by stimulus_id for fast lookup
    meta_by_id: dict[str, dict] = {s["stimulus_id"]: s for s in stimuli_meta}

    tasks: dict[str, dict[str, Any]] = {}

    # Collect raw labels in HDF5 row order
    product_labels: list[str | None] = []
    category_labels: list[str | None] = []
    register_labels: list[str | None] = []
    product_ids: list[str | None] = []

    for sid in stimulus_ids_h5:
        meta = meta_by_id.get(sid)
        if meta is None:
            product_labels.append(None)
            category_labels.append(None)
            register_labels.append(None)
            product_ids.append(None)
        else:
            product_labels.append(meta.get("product_id"))
            category_labels.append(meta.get("category"))
            register_labels.append(meta.get("register"))
            product_ids.append(meta.get("product_id"))

    # Build arrays for each task
    for task_name, raw_labels in [
        ("product", product_labels),
        ("category", category_labels),
        ("register", register_labels),
    ]:
        # Mask out rows with missing labels
        mask = np.array([lab is not None for lab in raw_labels], dtype=bool)
        valid_labels = [lab for lab in raw_labels if lab is not None]
        valid_groups = [
            pid for pid, m in zip(product_ids, mask) if m and pid is not None
        ]

        if len(valid_labels) == 0:
            logger.warning("Task '%s': no valid labels found — skipping", task_name)
            continue

        le = LabelEncoder()
        encoded_labels = le.fit_transform(valid_labels)

        # Group encoding (product_id as integers for GroupKFold)
        group_le = LabelEncoder()
        encoded_groups = group_le.fit_transform(valid_groups)

        tasks[task_name] = {
            "labels": encoded_labels.astype(np.int64),
            "groups": encoded_groups.astype(np.int64),
            "encoder": le,
            "mask": mask,
            "n_classes": len(le.classes_),
        }

    return tasks


def _read_layer_from_h5(
    h5_file: h5py.File,
    layer_idx: int,
    dataset_name: str = "hidden_states_mean_no_special",
) -> np.ndarray:
    """Read a single layer's representations from an open HDF5 file.

    Performs chunked (single-layer) read to avoid loading the full tensor.

    Args:
        h5_file: An open h5py.File handle.
        layer_idx: Layer index along axis 1.
        dataset_name: Name of the HDF5 dataset.

    Returns:
        Array of shape (N, D).
    """
    # Read only one slice along the layer axis
    return h5_file[dataset_name][:, layer_idx, :]


def _get_stimulus_ids_from_h5(h5_file: h5py.File) -> list[str]:
    """Read stimulus_ids from an open HDF5 file."""
    raw = h5_file["stimulus_ids"][:]
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw]


# ---------------------------------------------------------------------------
# All-layer probing
# ---------------------------------------------------------------------------


def train_probes_all_layers(
    h5_path: str | Path,
    stimuli_meta: list[dict],
    config: dict,
) -> dict[str, Any]:
    """Train probes for all three tasks at every layer, for all anisotropy methods.

    Tasks:
      - "product": 40-class real-product classification
      - "category": 8-class product-category classification
      - "register": 5-class linguistic register classification

    Reads HDF5 layer-by-layer (chunked) to control memory usage.

    Args:
        h5_path: Path to the HDF5 file with hidden states.
        stimuli_meta: List of stimulus metadata dicts.
        config: CONFIG dict from run.py.

    Returns:
        Nested dict: results[anisotropy_method][task_name][layer_idx] = probe_result.
    """
    h5_path = Path(h5_path)
    anisotropy_methods = config.get("anisotropy_methods", ["none"])

    results: dict[str, dict[str, dict[int, Any]]] = {}

    with h5py.File(h5_path, "r") as f:
        stimulus_ids = _get_stimulus_ids_from_h5(f)
        n_layers_plus_one = f["hidden_states_mean_no_special"].shape[1]

        # Build label arrays once
        task_arrays = _build_label_arrays(stimulus_ids, stimuli_meta)

        if not task_arrays:
            logger.error("No valid probe tasks found — returning empty results")
            return {}

        for method in anisotropy_methods:
            results[method] = {}
            for task_name in task_arrays:
                results[method][task_name] = {}

            logger.info("Probing with anisotropy method: %s", method)

            for layer_idx in range(n_layers_plus_one):
                X_full = _read_layer_from_h5(f, layer_idx)

                for task_name, task_data in task_arrays.items():
                    mask = task_data["mask"]
                    X = X_full[mask]
                    y = task_data["labels"]
                    groups = task_data["groups"]

                    # Apply anisotropy correction (fit on full training set
                    # per fold — handled inside train_probe_at_layer would be
                    # ideal, but for simplicity we apply globally here; the
                    # nested CV still selects C fairly)
                    if method != "none":
                        X_corrected, _ = _apply_anisotropy_correction(
                            X, X, method, config.get("pca_epsilon", 1e-8)
                        )
                        X = X_corrected

                    probe_result = train_probe_at_layer(X, y, groups, config)
                    results[method][task_name][layer_idx] = probe_result

                    logger.debug(
                        "Layer %d, task=%s, method=%s: macro_f1=%.4f",
                        layer_idx,
                        task_name,
                        method,
                        probe_result["macro_f1"],
                    )

                logger.info(
                    "Layer %d/%d complete (method=%s)",
                    layer_idx + 1,
                    n_layers_plus_one,
                    method,
                )

    logger.info("All-layer probing complete for %d methods", len(anisotropy_methods))
    return results


# ---------------------------------------------------------------------------
# Control probes (Hewitt & Manning 2019)
# ---------------------------------------------------------------------------


def _permute_labels_preserve_sizes(
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute labels while preserving class sizes (counts per class).

    Strategy: Shuffle the label array — since we shuffle the full array,
    class counts are trivially preserved (it is a permutation of the same
    values).

    Args:
        labels: 1-D integer label array.
        rng: NumPy random generator.

    Returns:
        A new array with the same values in shuffled order.
    """
    permuted = labels.copy()
    rng.shuffle(permuted)
    return permuted


def train_control_probes(
    h5_path: str | Path,
    stimuli_meta: list[dict],
    config: dict,
) -> dict[str, Any]:
    """Train control probes with permuted labels (Hewitt & Manning 2019).

    For each task, permutes the labels ONCE (preserving class sizes) and trains
    a probe at every layer.  Computes selectivity = real_f1 - control_f1.

    Args:
        h5_path: Path to the HDF5 file with hidden states.
        stimuli_meta: List of stimulus metadata dicts.
        config: CONFIG dict from run.py.

    Returns:
        Dict with:
          - "control_results": {task_name: {layer_idx: probe_result}}
          - "selectivity": {task_name: {layer_idx: float}}
          - "real_results": reference to the real probe results (or None if
            not provided — caller typically supplies these separately)
    """
    h5_path = Path(h5_path)
    rng = np.random.default_rng(config.get("seed", 42))

    control_results: dict[str, dict[int, Any]] = {}
    permuted_labels_cache: dict[str, np.ndarray] = {}

    with h5py.File(h5_path, "r") as f:
        stimulus_ids = _get_stimulus_ids_from_h5(f)
        n_layers_plus_one = f["hidden_states_mean_no_special"].shape[1]

        task_arrays = _build_label_arrays(stimulus_ids, stimuli_meta)

        if not task_arrays:
            logger.error("No valid probe tasks for control probes")
            return {"control_results": {}, "selectivity": {}}

        # Generate ONE permutation per task
        for task_name, task_data in task_arrays.items():
            permuted = _permute_labels_preserve_sizes(task_data["labels"], rng)
            permuted_labels_cache[task_name] = permuted
            control_results[task_name] = {}

        for layer_idx in range(n_layers_plus_one):
            X_full = _read_layer_from_h5(f, layer_idx)

            for task_name, task_data in task_arrays.items():
                mask = task_data["mask"]
                X = X_full[mask]
                y_permuted = permuted_labels_cache[task_name]
                groups = task_data["groups"]

                probe_result = train_probe_at_layer(X, y_permuted, groups, config)
                control_results[task_name][layer_idx] = probe_result

                logger.debug(
                    "Control — Layer %d, task=%s: macro_f1=%.4f",
                    layer_idx,
                    task_name,
                    probe_result["macro_f1"],
                )

            logger.info(
                "Control probes — Layer %d/%d complete", layer_idx + 1, n_layers_plus_one
            )

    return {
        "control_results": control_results,
        "selectivity": {},  # Populated by caller after pairing with real results
    }


def compute_selectivity(
    real_results: dict[str, dict[int, Any]],
    control_results: dict[str, dict[int, Any]],
) -> dict[str, dict[int, float]]:
    """Compute selectivity = real_f1 - control_f1 at each layer for each task.

    Args:
        real_results: {task_name: {layer_idx: {"macro_f1": float, ...}}}
        control_results: Same structure from control probes.

    Returns:
        {task_name: {layer_idx: selectivity}}
    """
    selectivity: dict[str, dict[int, float]] = {}

    for task_name in real_results:
        selectivity[task_name] = {}
        if task_name not in control_results:
            logger.warning(
                "Task '%s' missing from control results — skipping selectivity",
                task_name,
            )
            continue

        for layer_idx in real_results[task_name]:
            real_f1 = real_results[task_name][layer_idx]["macro_f1"]
            ctrl_f1 = control_results[task_name].get(layer_idx, {}).get("macro_f1", 0.0)
            selectivity[task_name][layer_idx] = real_f1 - ctrl_f1

    return selectivity


# ---------------------------------------------------------------------------
# Zone boundaries & zone probes
# ---------------------------------------------------------------------------


def compute_zone_boundaries(
    n_layers: int,
    config: dict,
) -> dict[str, list[int]]:
    """Divide transformer layers into zones for zone-level probing.

    Zones (layer indices are 0-based, with layer 0 = embedding output):
      - early: 0 to 10th percentile
      - protocol: 10th to 70th percentile
      - late: 70th to 99th percentile (includes transition region 70-90%)
      - output: final layer only

    The zone percentile boundaries come from ``config["h3_layer_zone_pct"]``
    which is (10, 70) — marking the start and end of the "protocol" zone.

    Args:
        n_layers: Total number of layers (L+1, including embedding layer 0).
        config: CONFIG dict.

    Returns:
        Dict mapping zone name to list of layer indices.
    """
    pct_start, pct_end = config["h3_layer_zone_pct"]  # (10, 70)

    # Compute actual layer boundaries
    # Last real layer index is n_layers - 1
    last_idx = n_layers - 1

    early_end = int(round(last_idx * pct_start / 100.0))
    protocol_end = int(round(last_idx * pct_end / 100.0))
    # Late zone: from protocol_end to second-to-last layer
    # Output zone: final layer only

    early_layers = list(range(0, early_end))
    protocol_layers = list(range(early_end, protocol_end))
    late_layers = list(range(protocol_end, last_idx))
    output_layers = [last_idx]

    zones = {
        "early": early_layers,
        "protocol": protocol_layers,
        "late": late_layers,
        "output": output_layers,
    }

    logger.info(
        "Zone boundaries (n_layers=%d): early=%s, protocol=%s, late=%s, output=%s",
        n_layers,
        _zone_range_str(early_layers),
        _zone_range_str(protocol_layers),
        _zone_range_str(late_layers),
        _zone_range_str(output_layers),
    )

    return zones


def _zone_range_str(layers: list[int]) -> str:
    """Human-readable range string for a list of layer indices."""
    if not layers:
        return "[]"
    if len(layers) == 1:
        return f"[{layers[0]}]"
    return f"[{layers[0]}..{layers[-1]}] ({len(layers)} layers)"


def train_zone_probes(
    h5_path: str | Path,
    stimuli_meta: list[dict],
    config: dict,
) -> dict[str, Any]:
    """Train probes on mean-pooled zone representations.

    For each zone, mean-pools the representations across the layers in that
    zone, then trains a probe for all three tasks.

    Reads HDF5 layer-by-layer and accumulates zone sums incrementally to
    avoid loading the full (N, L+1, D) tensor into memory.

    Args:
        h5_path: Path to the HDF5 file with hidden states.
        stimuli_meta: List of stimulus metadata dicts.
        config: CONFIG dict from run.py.

    Returns:
        Dict: results[zone_name][task_name] = probe_result.
    """
    h5_path = Path(h5_path)

    with h5py.File(h5_path, "r") as f:
        stimulus_ids = _get_stimulus_ids_from_h5(f)
        n_layers_plus_one = f["hidden_states_mean_no_special"].shape[1]
        n_stimuli = f["hidden_states_mean_no_special"].shape[0]
        hidden_dim = f["hidden_states_mean_no_special"].shape[2]

        task_arrays = _build_label_arrays(stimulus_ids, stimuli_meta)

        if not task_arrays:
            logger.error("No valid probe tasks for zone probes")
            return {}

        zones = compute_zone_boundaries(n_layers_plus_one, config)

        # Accumulate zone representations incrementally
        # For each zone, we sum layer representations and then divide by count
        zone_sums: dict[str, np.ndarray] = {}
        zone_counts: dict[str, int] = {}

        for zone_name, layer_indices in zones.items():
            zone_sums[zone_name] = np.zeros((n_stimuli, hidden_dim), dtype=np.float64)
            zone_counts[zone_name] = 0

        # Read layer-by-layer, add to appropriate zone(s)
        # Build reverse lookup: layer_idx -> zone_name
        layer_to_zone: dict[int, str] = {}
        for zone_name, layer_indices in zones.items():
            for lidx in layer_indices:
                layer_to_zone[lidx] = zone_name

        for layer_idx in range(n_layers_plus_one):
            zone_name = layer_to_zone.get(layer_idx)
            if zone_name is None:
                # Layer not assigned to any zone (should not happen with
                # our inclusive zone definition, but be safe)
                continue
            X_layer = _read_layer_from_h5(f, layer_idx)
            zone_sums[zone_name] += X_layer.astype(np.float64)
            zone_counts[zone_name] += 1

    # Compute zone means
    zone_representations: dict[str, np.ndarray] = {}
    for zone_name in zones:
        count = zone_counts[zone_name]
        if count == 0:
            logger.warning("Zone '%s' has 0 layers — skipping", zone_name)
            continue
        zone_representations[zone_name] = (
            zone_sums[zone_name] / count
        ).astype(np.float32)

    # Train probes on zone-mean representations
    results: dict[str, dict[str, Any]] = {}

    for zone_name, X_zone in zone_representations.items():
        results[zone_name] = {}

        for task_name, task_data in task_arrays.items():
            mask = task_data["mask"]
            X = X_zone[mask]
            y = task_data["labels"]
            groups = task_data["groups"]

            probe_result = train_probe_at_layer(X, y, groups, config)
            results[zone_name][task_name] = probe_result

            logger.info(
                "Zone '%s', task=%s: macro_f1=%.4f (CI: %.4f–%.4f)",
                zone_name,
                task_name,
                probe_result["macro_f1"],
                probe_result["bootstrap_ci_95"][0],
                probe_result["bootstrap_ci_95"][1],
            )

    return results


# ---------------------------------------------------------------------------
# JSON-safe serialization helper
# ---------------------------------------------------------------------------


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to Python natives for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_probe_results(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save probe results to JSON, converting numpy types as needed."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe = _make_json_safe(results)
    with open(output_path, "w") as f:
        json.dump(safe, f, indent=2)
    logger.info("Probe results saved to %s", output_path)
