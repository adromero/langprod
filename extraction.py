"""Extraction module — hidden-state extraction from transformer models.

Loads models, registers forward hooks on all transformer layers, and extracts
hidden states with mean-pooling (excluding special tokens).  Writes results
incrementally to HDF5 with gzip compression and supports resume.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_name: str,
    device_map: str = "auto",
) -> tuple[Any, Any]:
    """Load a causal-LM and its tokenizer.

    Handles:
      - GPTQ-4bit via native transformers (>=4.40) loading
      - FP16 with device_map="auto" (CPU-offload via accelerate)
      - Standard FP16

    Returns the model in eval() mode with pad_token set if missing.
    """
    logger.info("Loading model: %s (device_map=%s)", model_name, device_map)

    # Detect GPTQ by name convention
    name_lower = model_name.lower()
    is_gptq = "gptq" in name_lower

    load_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
    }

    if is_gptq:
        # Native transformers GPTQ — no extra library needed for >=4.40
        logger.info("Detected GPTQ model — loading via native transformers GPTQ integration")
    else:
        # Standard FP16
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token (%s)", tokenizer.eos_token)

    logger.info("Model loaded: %d parameters", sum(p.numel() for p in model.parameters()))
    return model, tokenizer


# ---------------------------------------------------------------------------
# Architecture helpers
# ---------------------------------------------------------------------------


def get_layer_count(model: Any) -> int:
    """Return the number of transformer layers (architecture-agnostic).

    Discovers layers by walking named_modules() and counting modules whose
    name matches the pattern ``model.layers.N`` or ``transformer.h.N`` etc.
    """
    layer_names = _discover_transformer_layer_names(model)
    count = len(layer_names)
    if count == 0:
        raise RuntimeError(
            "Could not discover transformer layers via named_modules(). "
            "Check model architecture."
        )
    return count


def get_hidden_dim(model: Any) -> int:
    """Return the hidden dimension of the model (architecture-agnostic)."""
    config = model.config
    for attr in ("hidden_size", "d_model", "n_embd"):
        if hasattr(config, attr):
            return getattr(config, attr)
    raise RuntimeError("Cannot determine hidden dimension from model.config")


def _discover_transformer_layer_names(model: Any) -> list[str]:
    """Return a sorted list of full names for top-level transformer layer modules.

    Works with Qwen, Llama, GPT-NeoX, Falcon, Mistral, Phi, etc.  The heuristic
    looks for modules whose name matches ``*.layers.N`` or ``*.h.N`` where N is
    an integer, and the module is NOT a sub-component (e.g., not self_attn).
    """
    # Pattern: something.layers.<int> or something.h.<int>
    pattern = re.compile(r"^(.+)\.(layers|h)\.(\d+)$")
    candidates: dict[str, Any] = {}
    for name, module in model.named_modules():
        m = pattern.match(name)
        if m:
            # Ensure this is a top-level layer, not a sub-module of a layer
            # (e.g., model.model.layers.0 but NOT model.model.layers.0.self_attn)
            candidates[name] = module
    # Sort by layer index
    def _sort_key(name: str) -> int:
        m = pattern.match(name)
        return int(m.group(3)) if m else 0

    return sorted(candidates.keys(), key=_sort_key)


def _discover_submodule(layer_module: Any, layer_name: str, suffix: str) -> tuple[str, Any] | None:
    """Find a named sub-module (e.g. self_attn, mlp) under a layer module."""
    for name, mod in layer_module.named_modules():
        if name == suffix:
            return f"{layer_name}.{suffix}", mod
    return None


# ---------------------------------------------------------------------------
# Extraction hooks
# ---------------------------------------------------------------------------


class ExtractionHooks:
    """Register forward hooks on all transformer layers to capture intermediate outputs.

    Captures:
      - attention output (from self_attn sub-module)
      - MLP output (from mlp sub-module)

    All captured tensors are immediately .detach().cpu()'d to avoid GPU memory
    buildup.
    """

    ATTN_SUFFIXES = ("self_attn", "attention", "attn")
    MLP_SUFFIXES = ("mlp", "feed_forward", "ffn")

    def __init__(self, model: Any) -> None:
        self.model = model
        self._handles: list[Any] = []
        self.attention_outputs: dict[int, torch.Tensor] = {}
        self.mlp_outputs: dict[int, torch.Tensor] = {}
        self._layer_names = _discover_transformer_layer_names(model)
        self._n_layers = len(self._layer_names)

        if self._n_layers == 0:
            raise RuntimeError("No transformer layers discovered for hook registration")

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Walk through discovered layers and attach hooks to attn and mlp sub-modules."""
        for layer_idx, layer_name in enumerate(self._layer_names):
            # Retrieve the actual module object
            layer_module = dict(self.model.named_modules())[layer_name]

            # Attention hook
            attn_registered = False
            for suffix in self.ATTN_SUFFIXES:
                result = _discover_submodule(layer_module, layer_name, suffix)
                if result is not None:
                    _, attn_mod = result
                    handle = attn_mod.register_forward_hook(
                        self._make_hook("attention", layer_idx)
                    )
                    self._handles.append(handle)
                    attn_registered = True
                    break
            if not attn_registered:
                logger.warning(
                    "Layer %d (%s): no attention sub-module found", layer_idx, layer_name
                )

            # MLP hook
            mlp_registered = False
            for suffix in self.MLP_SUFFIXES:
                result = _discover_submodule(layer_module, layer_name, suffix)
                if result is not None:
                    _, mlp_mod = result
                    handle = mlp_mod.register_forward_hook(
                        self._make_hook("mlp", layer_idx)
                    )
                    self._handles.append(handle)
                    mlp_registered = True
                    break
            if not mlp_registered:
                logger.warning(
                    "Layer %d (%s): no MLP sub-module found", layer_idx, layer_name
                )

    def _make_hook(self, kind: str, layer_idx: int):
        """Return a hook function that captures the module output."""
        def hook_fn(module, input, output):
            # output may be a tuple — take the first element (the hidden state tensor)
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output
            captured = tensor.detach().cpu()
            if kind == "attention":
                self.attention_outputs[layer_idx] = captured
            else:
                self.mlp_outputs[layer_idx] = captured
        return hook_fn

    def clear(self) -> None:
        """Clear all captured outputs."""
        self.attention_outputs.clear()
        self.mlp_outputs.clear()

    def remove(self) -> None:
        """Remove all registered hooks from the model."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def n_layers(self) -> int:
        return self._n_layers


# ---------------------------------------------------------------------------
# Pooling functions
# ---------------------------------------------------------------------------


def mean_pool_no_special(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool hidden states excluding BOS, EOS, and PAD tokens.

    Args:
        hidden_states: shape (batch, seq_len, hidden_dim)
        attention_mask: shape (batch, seq_len) — 1 for real tokens, 0 for padding
        special_ids: set of token IDs to exclude (BOS, EOS, PAD)
        input_ids: shape (batch, seq_len) — token IDs

    Returns:
        Pooled tensor of shape (batch, hidden_dim).
    """
    # Build mask: 1 where token is non-special AND not padding
    special_mask = torch.zeros_like(attention_mask, dtype=torch.float32)
    for sid in special_ids:
        special_mask += (input_ids == sid).float()
    # valid = attended AND not special
    valid_mask = attention_mask.float() * (1.0 - special_mask.clamp(max=1.0))

    # Expand for broadcasting: (batch, seq_len, 1)
    valid_mask_expanded = valid_mask.unsqueeze(-1)

    # Sum of valid hidden states
    summed = (hidden_states.float() * valid_mask_expanded).sum(dim=1)
    counts = valid_mask_expanded.sum(dim=1).clamp(min=1.0)

    return (summed / counts).squeeze(0)  # (hidden_dim,) if batch=1


def last_token_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool by taking the last non-padding token's hidden state.

    Args:
        hidden_states: shape (batch, seq_len, hidden_dim)
        attention_mask: shape (batch, seq_len)

    Returns:
        Tensor of shape (batch, hidden_dim).
    """
    # Find index of last attended token for each sequence in the batch
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    pooled = hidden_states[batch_indices, seq_lengths.long()]
    return pooled.squeeze(0)  # (hidden_dim,) if batch=1


# ---------------------------------------------------------------------------
# NaN / Inf checking
# ---------------------------------------------------------------------------


def _check_nan_inf(tensor: torch.Tensor | np.ndarray, label: str) -> bool:
    """Return True if tensor contains NaN or Inf. Logs a warning if so."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.float()
        has_nan = torch.isnan(arr).any().item()
        has_inf = torch.isinf(arr).any().item()
    else:
        has_nan = bool(np.isnan(tensor).any())
        has_inf = bool(np.isinf(tensor).any())

    if has_nan or has_inf:
        logger.warning("NaN/Inf detected in %s (nan=%s, inf=%s)", label, has_nan, has_inf)
        return True
    return False


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def _sanitize_model_name(model_name: str) -> str:
    """Convert a HuggingFace model name to a filesystem-safe string."""
    return model_name.replace("/", "_").replace("\\", "_")


def _get_h5_path(config: dict, model_name: str) -> Path:
    """Return the HDF5 output path for a given model."""
    output_dir = Path(config["output_dir"])
    sanitized = _sanitize_model_name(model_name)
    return output_dir / f"{sanitized}_hidden_states.h5"


def _get_existing_stimulus_ids(h5_path: Path) -> set[str]:
    """Return the set of stimulus_ids already written to an HDF5 file."""
    if not h5_path.exists():
        return set()
    try:
        with h5py.File(h5_path, "r") as f:
            if "stimulus_ids" in f:
                return set(s.decode("utf-8") if isinstance(s, bytes) else s for s in f["stimulus_ids"][:])
    except Exception as e:
        logger.warning("Could not read existing HDF5 for resume check: %s", e)
    return set()


def _init_h5_datasets(
    h5_path: Path,
    n_stimuli: int,
    n_layers: int,
    hidden_dim: int,
) -> None:
    """Create or verify HDF5 datasets with the correct shapes.

    Datasets are created with maxshape allowing unlimited growth along axis 0
    so that we can resize as we append rows.
    """
    with h5py.File(h5_path, "a") as f:
        # hidden_states: (N, L+1, D) — embedding layer + L transformer layers
        if "hidden_states_mean_no_special" not in f:
            f.create_dataset(
                "hidden_states_mean_no_special",
                shape=(0, n_layers + 1, hidden_dim),
                maxshape=(None, n_layers + 1, hidden_dim),
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
        # attention output: (N, L, D)
        if "attention_output_mean_no_special" not in f:
            f.create_dataset(
                "attention_output_mean_no_special",
                shape=(0, n_layers, hidden_dim),
                maxshape=(None, n_layers, hidden_dim),
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
        # MLP output: (N, L, D)
        if "mlp_output_mean_no_special" not in f:
            f.create_dataset(
                "mlp_output_mean_no_special",
                shape=(0, n_layers, hidden_dim),
                maxshape=(None, n_layers, hidden_dim),
                dtype="float32",
                compression="gzip",
                compression_opts=4,
            )
        # stimulus_ids: variable-length strings
        if "stimulus_ids" not in f:
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset(
                "stimulus_ids",
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
            )


def _append_to_h5(
    h5_path: Path,
    stimulus_id: str,
    hidden_states_row: np.ndarray,
    attn_row: np.ndarray,
    mlp_row: np.ndarray,
) -> None:
    """Append one stimulus's results to the HDF5 file.

    Each row is appended by resizing the dataset along axis 0.
    """
    with h5py.File(h5_path, "a") as f:
        for ds_name, row in [
            ("hidden_states_mean_no_special", hidden_states_row),
            ("attention_output_mean_no_special", attn_row),
            ("mlp_output_mean_no_special", mlp_row),
        ]:
            ds = f[ds_name]
            n = ds.shape[0]
            ds.resize(n + 1, axis=0)
            ds[n] = row

        # Append stimulus_id
        ids_ds = f["stimulus_ids"]
        n = ids_ds.shape[0]
        ids_ds.resize(n + 1, axis=0)
        ids_ds[n] = stimulus_id


# ---------------------------------------------------------------------------
# Extraction pipeline
# ---------------------------------------------------------------------------


def extract_hidden_states(
    config: dict,
    stimuli: list[dict],
    model_name: str,
) -> Path:
    """Extract hidden states from all stimuli through the specified model.

    Processes stimuli one at a time (batch_size=1).  For each stimulus:
      1. Tokenize the text
      2. Forward pass with output_hidden_states=True and hooks attached
      3. Mean-pool each layer's hidden states (excluding special tokens)
      4. Check for NaN/Inf — skip stimulus if detected
      5. Append results incrementally to HDF5

    Supports resume: stimuli already present in the HDF5 file are skipped.

    Args:
        config: The CONFIG dict from run.py.
        stimuli: List of stimulus dicts (each must have 'stimulus_id' and 'text').
        model_name: HuggingFace model identifier.

    Returns:
        Path to the output HDF5 file.
    """
    h5_path = _get_h5_path(config, model_name)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: find already-processed stimulus IDs
    existing_ids = _get_existing_stimulus_ids(h5_path)
    stimuli_to_process = [s for s in stimuli if s["stimulus_id"] not in existing_ids]

    if existing_ids:
        logger.info(
            "Resume: %d stimuli already in HDF5, %d remaining",
            len(existing_ids),
            len(stimuli_to_process),
        )

    if not stimuli_to_process:
        logger.info("All stimuli already processed — nothing to do.")
        return h5_path

    # Load model and tokenizer
    device_map = "auto"
    model, tokenizer = load_model_and_tokenizer(model_name, device_map=device_map)

    n_layers = get_layer_count(model)
    hidden_dim = get_hidden_dim(model)
    logger.info("Model has %d layers, hidden_dim=%d", n_layers, hidden_dim)

    # Build set of special token IDs to exclude from mean pooling
    special_ids: set[int] = set()
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)

    # Initialize HDF5 datasets
    _init_h5_datasets(h5_path, n_stimuli=len(stimuli), n_layers=n_layers, hidden_dim=hidden_dim)

    # Register hooks
    hooks = ExtractionHooks(model)
    logger.info("Registered hooks on %d layers", hooks.n_layers)

    skipped = 0
    processed = 0

    try:
        for stim in tqdm(stimuli_to_process, desc=f"Extracting ({model_name})"):
            stimulus_id = stim["stimulus_id"]
            text = stim["text"]

            try:
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=False,
                    truncation=False,
                )
                # Move to model's device
                device = next(model.parameters()).device
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                # Forward pass
                hooks.clear()
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )

                # -------------------------------------------------------
                # Collect hidden states from output_hidden_states
                # Shape of each: (1, seq_len, hidden_dim)
                # outputs.hidden_states is a tuple of (L+1) tensors:
                #   [embedding_output, layer_0_output, ..., layer_{L-1}_output]
                # -------------------------------------------------------
                all_hidden = outputs.hidden_states  # tuple of (1, seq, D)

                # Pool each layer: (L+1,) entries -> each pooled to (D,)
                input_ids_cpu = input_ids.detach().cpu()
                attn_mask_cpu = attention_mask.detach().cpu()

                hidden_pooled = []
                for layer_hs in all_hidden:
                    hs_cpu = layer_hs.detach().cpu()
                    pooled = mean_pool_no_special(hs_cpu, attn_mask_cpu, special_ids, input_ids_cpu)
                    hidden_pooled.append(pooled.numpy())

                hidden_states_row = np.stack(hidden_pooled, axis=0)  # (L+1, D)

                # Pool attention and MLP hook outputs: (L,) entries
                attn_pooled = []
                mlp_pooled = []
                for layer_idx in range(n_layers):
                    # Attention
                    if layer_idx in hooks.attention_outputs:
                        a = hooks.attention_outputs[layer_idx]
                        a_pooled = mean_pool_no_special(a, attn_mask_cpu, special_ids, input_ids_cpu)
                        attn_pooled.append(a_pooled.numpy())
                    else:
                        # Fallback: zeros
                        logger.warning("Missing attention output for layer %d, stimulus %s", layer_idx, stimulus_id)
                        attn_pooled.append(np.zeros(hidden_dim, dtype=np.float32))

                    # MLP
                    if layer_idx in hooks.mlp_outputs:
                        m = hooks.mlp_outputs[layer_idx]
                        m_pooled = mean_pool_no_special(m, attn_mask_cpu, special_ids, input_ids_cpu)
                        mlp_pooled.append(m_pooled.numpy())
                    else:
                        logger.warning("Missing MLP output for layer %d, stimulus %s", layer_idx, stimulus_id)
                        mlp_pooled.append(np.zeros(hidden_dim, dtype=np.float32))

                attn_row = np.stack(attn_pooled, axis=0)  # (L, D)
                mlp_row = np.stack(mlp_pooled, axis=0)    # (L, D)

                # NaN/Inf check — skip this stimulus if bad
                if (
                    _check_nan_inf(hidden_states_row, f"hidden_states/{stimulus_id}")
                    or _check_nan_inf(attn_row, f"attn_output/{stimulus_id}")
                    or _check_nan_inf(mlp_row, f"mlp_output/{stimulus_id}")
                ):
                    logger.warning("Skipping stimulus %s due to NaN/Inf", stimulus_id)
                    skipped += 1
                    continue

                # Write to HDF5
                _append_to_h5(h5_path, stimulus_id, hidden_states_row, attn_row, mlp_row)
                processed += 1

            except torch.cuda.OutOfMemoryError:
                logger.error("CUDA OOM on stimulus %s — skipping", stimulus_id)
                torch.cuda.empty_cache()
                skipped += 1
                continue

            finally:
                # GPU memory cleanup after each forward pass
                hooks.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    finally:
        hooks.remove()

    logger.info(
        "Extraction complete: %d processed, %d skipped, %d previously done",
        processed, skipped, len(existing_ids),
    )
    return h5_path


# ---------------------------------------------------------------------------
# Pilot validation
# ---------------------------------------------------------------------------


def run_pilot(config: dict, n: int = 5) -> dict:
    """Run a 5-stimulus pilot validation to verify extraction works end-to-end.

    Checks:
      (a) Model loads correctly
      (b) output_hidden_states returns correct layer count and shapes
      (c) Forward hooks fire on all layers
      (d) No NaN/Inf in captured tensors
      (e) Wall-clock time per stimulus

    Args:
        config: The CONFIG dict from run.py.
        n: Number of stimuli to test (default 5).

    Returns:
        Dict with validation results.
    """
    stimuli_path = Path(config["output_dir"]) / "stimuli.json"
    if not stimuli_path.exists():
        raise FileNotFoundError(f"Stimuli file not found: {stimuli_path}")

    with open(stimuli_path) as f:
        all_stimuli = json.load(f)

    pilot_stimuli = all_stimuli[:n]
    model_name = config["primary_model"]

    logger.info("=== PILOT VALIDATION: %d stimuli with %s ===", n, model_name)

    # (a) Load model
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(model_name)
    load_time = time.time() - t0
    logger.info("(a) Model loaded in %.1f seconds", load_time)

    n_layers = get_layer_count(model)
    hidden_dim = get_hidden_dim(model)

    # Build special IDs
    special_ids: set[int] = set()
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)

    # Register hooks
    hooks = ExtractionHooks(model)

    results = {
        "model_name": model_name,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "load_time_s": load_time,
        "hooks_registered": hooks.n_layers,
        "stimuli_tested": [],
        "all_passed": True,
    }

    try:
        for stim in pilot_stimuli:
            stim_result: dict[str, Any] = {"stimulus_id": stim["stimulus_id"]}
            text = stim["text"]

            inputs = tokenizer(text, return_tensors="pt", padding=False, truncation=False)
            device = next(model.parameters()).device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            hooks.clear()
            t_start = time.time()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            t_elapsed = time.time() - t_start

            # (b) Check hidden states shape
            all_hidden = outputs.hidden_states
            n_hidden = len(all_hidden)
            expected_n_hidden = n_layers + 1  # embedding + L layers
            shape_ok = n_hidden == expected_n_hidden
            stim_result["n_hidden_states"] = n_hidden
            stim_result["expected_n_hidden"] = expected_n_hidden
            stim_result["shape_ok"] = shape_ok
            if not shape_ok:
                logger.error(
                    "(b) FAIL: expected %d hidden states, got %d",
                    expected_n_hidden, n_hidden,
                )
                results["all_passed"] = False

            # (c) Check hooks fired
            attn_fired = len(hooks.attention_outputs)
            mlp_fired = len(hooks.mlp_outputs)
            hooks_ok = attn_fired == n_layers and mlp_fired == n_layers
            stim_result["attn_hooks_fired"] = attn_fired
            stim_result["mlp_hooks_fired"] = mlp_fired
            stim_result["hooks_ok"] = hooks_ok
            if not hooks_ok:
                logger.error(
                    "(c) FAIL: expected %d hooks, got attn=%d, mlp=%d",
                    n_layers, attn_fired, mlp_fired,
                )
                results["all_passed"] = False

            # (d) NaN/Inf check on all captured tensors
            nan_inf_found = False
            for i, hs in enumerate(all_hidden):
                hs_cpu = hs.detach().cpu()
                if _check_nan_inf(hs_cpu, f"hidden_state_layer_{i}"):
                    nan_inf_found = True

            for idx, t in hooks.attention_outputs.items():
                if _check_nan_inf(t, f"attn_output_layer_{idx}"):
                    nan_inf_found = True

            for idx, t in hooks.mlp_outputs.items():
                if _check_nan_inf(t, f"mlp_output_layer_{idx}"):
                    nan_inf_found = True

            # Also check pooled output
            input_ids_cpu = input_ids.detach().cpu()
            attn_mask_cpu = attention_mask.detach().cpu()
            for i, hs in enumerate(all_hidden):
                pooled = mean_pool_no_special(hs.detach().cpu(), attn_mask_cpu, special_ids, input_ids_cpu)
                if _check_nan_inf(pooled, f"pooled_hidden_layer_{i}"):
                    nan_inf_found = True

            stim_result["nan_inf_found"] = nan_inf_found
            if nan_inf_found:
                logger.error("(d) FAIL: NaN/Inf detected in stimulus %s", stim["stimulus_id"])
                results["all_passed"] = False

            # (e) Wall-clock time
            stim_result["time_s"] = t_elapsed
            logger.info(
                "Stimulus %s: %.2fs, shape_ok=%s, hooks_ok=%s, nan_inf=%s",
                stim["stimulus_id"], t_elapsed, shape_ok, hooks_ok, nan_inf_found,
            )

            results["stimuli_tested"].append(stim_result)

            # Cleanup
            hooks.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        hooks.remove()

    if results["all_passed"]:
        logger.info("=== PILOT VALIDATION PASSED ===")
    else:
        logger.error("=== PILOT VALIDATION FAILED ===")

    return results
