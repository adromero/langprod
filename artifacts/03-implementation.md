# Implementation Plan: The Protocol Layer Hypothesis

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Configuration System](#2-configuration-system)
3. [Stage 0: Project Scaffolding](#3-stage-0-project-scaffolding)
4. [Stage 1: Stimulus Generation](#4-stage-1-stimulus-generation)
5. [Stage 2: Hidden State Extraction](#5-stage-2-hidden-state-extraction)
6. [Stage 3: Similarity Analysis (RSA)](#6-stage-3-similarity-analysis)
7. [Stage 4: Linear Probes](#7-stage-4-linear-probes)
8. [Stage 5: Analysis & Reporting](#8-stage-5-analysis--reporting)
9. [Testing Strategy](#9-testing-strategy)
10. [Dependency & Execution Order](#10-dependency--execution-order)

---

## 1. Project Structure

```
protocol-layer-hypothesis/
├── pyproject.toml                  # Project metadata, dependencies, entry points
├── .env.example                    # API key placeholders
├── .env                            # (gitignored) Actual API keys
├── .gitignore
├── config/
│   ├── default.yaml                # Default experiment configuration
│   ├── debug.yaml                  # Small-scale debug run (2 products, 2 registers)
│   └── full.yaml                   # Full experiment configuration
├── src/
│   └── plh/                        # Package: protocol-layer-hypothesis
│       ├── __init__.py
│       ├── config.py               # Config loading and validation (Pydantic)
│       ├── constants.py            # Product definitions, category/register enums
│       ├── stage1_stimuli/
│       │   ├── __init__.py
│       │   ├── generate.py         # Stimulus generation orchestrator
│       │   ├── prompts.py          # Prompt templates for each register
│       │   ├── validate.py         # Semantic anchoring validation
│       │   └── schema.py           # Stimulus data models (Pydantic)
│       ├── stage2_extraction/
│       │   ├── __init__.py
│       │   ├── extract.py          # Hidden state extraction pipeline
│       │   ├── models.py           # Model loading (quantized, FP16, Llama)
│       │   ├── hooks.py            # Forward hooks for attention/MLP/residual decomposition
│       │   └── pooling.py          # Pooling strategies (mean, last-token)
│       ├── stage3_similarity/
│       │   ├── __init__.py
│       │   ├── rdm.py              # Representational Dissimilarity Matrices
│       │   ├── rsa.py              # RSA computation and statistical tests
│       │   ├── anisotropy.py       # Anisotropy correction (centering + whitening)
│       │   └── cosine.py           # Per-layer cosine similarity matrices
│       ├── stage4_probes/
│       │   ├── __init__.py
│       │   ├── train.py            # Probe training pipeline
│       │   ├── evaluate.py         # Evaluation with bootstrap CIs
│       │   └── zone_classifier.py  # Zone-based classification comparison
│       ├── stage5_analysis/
│       │   ├── __init__.py
│       │   ├── hypothesis_tests.py # Pre-registered hypothesis tests
│       │   ├── controls.py         # Memorization, quantization, generator controls
│       │   ├── reporting.py        # Summary report generation
│       │   └── go_no_go.py         # Go/no-go decision logic
│       ├── visualization/
│       │   ├── __init__.py
│       │   ├── phase_plots.py      # Phase structure plots
│       │   ├── probe_curves.py     # Probe accuracy across layers
│       │   ├── rsa_heatmaps.py     # RSA heatmaps
│       │   └── style.py            # Shared matplotlib styling
│       └── utils/
│           ├── __init__.py
│           ├── io.py               # HDF5/NPY I/O helpers
│           ├── seeds.py            # Reproducibility (seed management)
│           └── checkpoint.py       # Stage-level checkpointing
├── scripts/
│   ├── run_stage1.py               # CLI entry point for Stage 1
│   ├── run_stage2.py               # CLI entry point for Stage 2
│   ├── run_stage3.py               # CLI entry point for Stage 3
│   ├── run_stage4.py               # CLI entry point for Stage 4
│   ├── run_stage5.py               # CLI entry point for Stage 5
│   ├── run_all.py                  # Sequential pipeline runner
│   └── validate_data.py            # Inter-stage data validation
├── tests/
│   ├── conftest.py                 # Shared fixtures (mock stimuli, tiny hidden states)
│   ├── test_config.py
│   ├── test_stage1/
│   │   ├── test_prompts.py
│   │   ├── test_schema.py
│   │   └── test_validate.py
│   ├── test_stage2/
│   │   ├── test_hooks.py
│   │   ├── test_pooling.py
│   │   └── test_extract.py
│   ├── test_stage3/
│   │   ├── test_rdm.py
│   │   ├── test_rsa.py
│   │   └── test_anisotropy.py
│   ├── test_stage4/
│   │   ├── test_train.py
│   │   └── test_evaluate.py
│   └── test_stage5/
│       ├── test_hypothesis_tests.py
│       └── test_controls.py
├── data/                           # (gitignored except schemas)
│   ├── .gitkeep
│   ├── stimuli/                    # Stage 1 output
│   ├── hidden_states/              # Stage 2 output (HDF5)
│   ├── similarity/                 # Stage 3 output
│   ├── probes/                     # Stage 4 output
│   └── reports/                    # Stage 5 output
└── notebooks/                      # Optional exploratory notebooks
    └── .gitkeep
```

---

## 2. Configuration System

**Approach**: YAML config files + Pydantic validation + CLI overrides via argparse. Each script loads config from YAML, validates with Pydantic, and allows CLI overrides for key parameters.

### File: `config/default.yaml`

```yaml
experiment:
  name: "protocol-layer-hypothesis"
  seed: 42
  output_dir: "data"

stimuli:
  n_categories: 8
  n_products_per_category: 5  # 40 real products
  n_fictional_per_category: 5  # 40 fictional products
  registers: ["marketing", "regulatory", "social", "patent", "journalistic"]
  variants_per_register: 2  # 10 variants per product total
  target_token_range: [80, 150]
  core_attributes_per_product: [3, 5]  # min, max
  generators:
    primary: "anthropic"  # Claude for all 80 products
    cross_validation_subset_size: 10  # 10 products also generated by GPT-4 + human
    cross_validation_generators: ["openai", "human"]

models:
  primary:
    name: "Qwen/Qwen2.5-27B"  # Qwen3.5-27B -- use actual HF repo name
    quantization: "gptq-4bit"
    device_map: "auto"
    batch_size: 4  # Stimuli per forward pass
  fp16_subset:
    name: "Qwen/Qwen2.5-27B"
    quantization: null  # FP16
    device_map: "auto"  # Will use CPU offloading
    batch_size: 1
    stimulus_subset_fraction: 0.25  # 200 of 800 stimuli
  validation:
    name: "meta-llama/Llama-3.1-8B-Instruct"
    quantization: null
    device_map: "auto"
    batch_size: 8

extraction:
  components: ["hidden_states", "attention_output", "mlp_output", "residual_stream"]
  pooling: ["mean_no_special", "last_token"]
  save_format: "hdf5"
  checkpoint_every: 50  # Save checkpoint every 50 stimuli

similarity:
  anisotropy_correction: ["none", "mean_centering", "whitening"]
  model_rdm:
    same_product: 0.0
    same_category: 0.5
    different_category: 1.0

probes:
  tasks:
    - name: "product_40class"
      n_classes: 40
      primary: true
    - name: "category_8class"
      n_classes: 8
    - name: "register_5class"
      n_classes: 5
  regularization: 1.0  # L2 C parameter
  cv_folds: 5
  cv_stratify_by: ["category", "register"]
  metrics: ["macro_f1"]
  bootstrap_ci: 0.95
  bootstrap_n: 1000

zones:
  early: [0, 5]  # Layer indices, inclusive
  # protocol and late are computed as percentages of total layers
  protocol_pct: [0.10, 0.70]  # 10% to 70% of layer stack
  late_pct: [0.90, 0.99]  # Last 10% (minus output)
  # output = final layer

analysis:
  quantization_spearman_threshold: 0.9
  h1_significance: 0.05
  h2_register_dominance_margin: 5  # percentage points
  h3_protocol_advantage_margin: 2  # percentage points
```

### File: `src/plh/config.py`

```python
"""Configuration loading and validation."""

from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, field_validator

class StimuliConfig(BaseModel):
    n_categories: int = 8
    n_products_per_category: int = 5
    n_fictional_per_category: int = 5
    registers: list[str]
    variants_per_register: int = 2
    target_token_range: tuple[int, int] = (80, 150)
    core_attributes_per_product: tuple[int, int] = (3, 5)
    generators: dict

class ModelSpec(BaseModel):
    name: str
    quantization: Optional[str] = None
    device_map: str = "auto"
    batch_size: int = 4
    stimulus_subset_fraction: Optional[float] = None

class ModelsConfig(BaseModel):
    primary: ModelSpec
    fp16_subset: ModelSpec
    validation: ModelSpec

class ExtractionConfig(BaseModel):
    components: list[str]
    pooling: list[str]
    save_format: str = "hdf5"
    checkpoint_every: int = 50

class SimilarityConfig(BaseModel):
    anisotropy_correction: list[str]
    model_rdm: dict[str, float]

class ProbeTask(BaseModel):
    name: str
    n_classes: int
    primary: bool = False

class ProbesConfig(BaseModel):
    tasks: list[ProbeTask]
    regularization: float = 1.0
    cv_folds: int = 5
    cv_stratify_by: list[str]
    metrics: list[str]
    bootstrap_ci: float = 0.95
    bootstrap_n: int = 1000

class ZonesConfig(BaseModel):
    early: list[int]
    protocol_pct: list[float]
    late_pct: list[float]

class AnalysisConfig(BaseModel):
    quantization_spearman_threshold: float = 0.9
    h1_significance: float = 0.05
    h2_register_dominance_margin: float = 5.0
    h3_protocol_advantage_margin: float = 2.0

class ExperimentConfig(BaseModel):
    name: str
    seed: int
    output_dir: str

class Config(BaseModel):
    experiment: ExperimentConfig
    stimuli: StimuliConfig
    models: ModelsConfig
    extraction: ExtractionConfig
    similarity: SimilarityConfig
    probes: ProbesConfig
    zones: ZonesConfig
    analysis: AnalysisConfig

def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load config from YAML, apply overrides, validate."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if overrides:
        _deep_merge(raw, overrides)
    return Config(**raw)

def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base
```

---

## 3. Stage 0: Project Scaffolding

### Step 0.1: Initialize Project

**Files to create:**
- `pyproject.toml`
- `.gitignore`
- `.env.example`
- `src/plh/__init__.py`
- `src/plh/config.py` (as above)
- `src/plh/constants.py`
- `config/default.yaml` (as above)
- `config/debug.yaml`

**`pyproject.toml` key contents:**

```toml
[project]
name = "protocol-layer-hypothesis"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "transformers>=4.40",
    "accelerate>=0.28",
    "auto-gptq>=0.7",        # For GPTQ quantization
    "autoawq>=0.2",           # For AWQ quantization (alternative)
    "safetensors",
    "anthropic>=0.40",
    "openai>=1.12",
    "scikit-learn>=1.4",
    "numpy>=1.26",
    "scipy>=1.12",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "h5py>=3.10",
    "pyyaml>=6.0",
    "pydantic>=2.6",
    "tqdm>=4.66",
    "pandas>=2.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.0",
    "ruff>=0.3",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: marks tests requiring API keys or GPU (deselect with '-m \"not integration\"')",
]

[tool.ruff]
line-length = 100
```

**`.env.example`:**

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
OPENAI_API_KEY=sk-your-key-here
```

**`.gitignore` must include:**

```
.env
data/stimuli/
data/hidden_states/
data/similarity/
data/probes/
data/reports/
*.h5
*.hdf5
*.npy
*.npz
__pycache__/
.ruff_cache/
```

### Step 0.2: Define Product Catalog

**File: `src/plh/constants.py`**

This file defines all 40 real products, 40 fictional products, their categories, core attributes, and register definitions. This is the single source of truth for the experiment's material.

```python
"""Product catalog, categories, registers, and core attributes.

This is the single source of truth for experiment materials.
All 40 real products and 40 fictional products are defined here
with their core semantic attributes that must appear in every variant.
"""

from dataclasses import dataclass, field
from enum import Enum


class Category(str, Enum):
    ORAL_CARE = "oral_care"
    PET_FOOD = "pet_food"
    HOME_CLEANING = "home_cleaning"
    SPORTS_NUTRITION = "sports_nutrition"
    BABY_CARE = "baby_care"
    COFFEE_BEVERAGE = "coffee_beverage"
    SKINCARE = "skincare"
    SMART_HOME = "smart_home"


class Register(str, Enum):
    MARKETING = "marketing"
    REGULATORY = "regulatory"
    SOCIAL = "social"
    PATENT = "patent"
    JOURNALISTIC = "journalistic"


@dataclass
class Product:
    id: str                        # e.g., "oral_care_real_01"
    name: str                      # e.g., "Crest Pro-Health Advanced"
    category: Category
    is_fictional: bool
    core_attributes: list[str]     # 3-5 factual claims that MUST appear in every variant
    distinguishing_features: list[str]  # Features unique to THIS product vs. category peers


# ---- Real Products (8 categories x 5 products = 40) ----
# NOTE: The implementing agent should fill in the full catalog.
# Below is the STRUCTURE with 2 examples per category to show the pattern.
# The implementing agent must expand this to 5 per category.

REAL_PRODUCTS: list[Product] = [
    # --- Oral Care ---
    Product(
        id="oral_care_real_01",
        name="Crest Pro-Health Advanced Electric Toothbrush",
        category=Category.ORAL_CARE,
        is_fictional=False,
        core_attributes=[
            "40,000 brush strokes per minute",
            "pressure sensor that pauses motor on excessive force",
            "2-minute timer with 30-second quadrant alerts",
            "rechargeable lithium battery, 14-day life per charge",
        ],
        distinguishing_features=["oscillating-rotating head", "Bluetooth app connectivity"],
    ),
    Product(
        id="oral_care_real_02",
        name="Lumineux Whitening Strips",
        category=Category.ORAL_CARE,
        is_fictional=False,
        core_attributes=[
            "certified non-toxic whitening using dead sea salt and coconut oil",
            "21-strip treatment course over 7 days",
            "enamel-safe formulation without hydrogen peroxide",
            "clinically tested to whiten 3 shades in 7 days",
        ],
        distinguishing_features=["peroxide-free", "dentist-developed"],
    ),
    # ... (3 more oral care real products)
    # ... (repeat for all 8 categories)
]


# ---- Fictional Products (8 categories x 5 products = 40) ----
# Fictional products must:
# 1. Have plausible but non-existent brand names
# 2. Have plausible but novel feature combinations
# 3. NOT replicate any real product's exact feature set
# 4. Have core attributes of comparable specificity to real products

FICTIONAL_PRODUCTS: list[Product] = [
    Product(
        id="oral_care_fict_01",
        name="AquaPulse Sonic Irrigating Toothbrush",
        category=Category.ORAL_CARE,
        is_fictional=True,
        core_attributes=[
            "integrated water flosser with 0.6mm jet nozzle",
            "ultrasonic vibration at 96,000 pulses per minute",
            "UV-C sterilization dock with 5-minute cycle",
            "built-in pH sensor that detects acidic oral conditions",
        ],
        distinguishing_features=["combined brushing and irrigation", "pH monitoring"],
    ),
    # ... (4 more oral care fictional, then all other categories)
]


# ---- Register Specifications ----
# These guide stimulus generation prompts.

REGISTER_SPECS: dict[Register, dict] = {
    Register.MARKETING: {
        "voice": "second person ('you')",
        "tone": "aspirational, benefit-led, emotional",
        "structure": "headline + body, bullet points optional",
        "vocabulary": "sensory, superlative, brand-forward",
        "example_source": "Amazon product listing, brand website",
    },
    Register.REGULATORY: {
        "voice": "passive, impersonal",
        "tone": "formal, precise, compliance-oriented",
        "structure": "numbered sections, specification tables",
        "vocabulary": "technical measurements, regulatory terminology",
        "example_source": "FDA filing, MSDS sheet, CPSC notice",
    },
    Register.SOCIAL: {
        "voice": "first person ('I', 'my')",
        "tone": "informal, conversational, opinionated",
        "structure": "stream-of-consciousness, fragments, emoji ok",
        "vocabulary": "slang, abbreviations, hedging",
        "example_source": "Reddit review, tweet thread, TikTok caption",
    },
    Register.PATENT: {
        "voice": "impersonal, third person",
        "tone": "maximally precise, defensive",
        "structure": "single long claim sentence, nested clauses",
        "vocabulary": "dense nominal phrases, 'comprising', 'wherein'",
        "example_source": "USPTO patent abstract, claims section",
    },
    Register.JOURNALISTIC: {
        "voice": "third person",
        "tone": "balanced, analytical, source-quoting",
        "structure": "inverted pyramid, expert quotes",
        "vocabulary": "neutral, attribution-heavy",
        "example_source": "Trade publication, consumer report",
    },
}


# Products selected for the cross-generator validation subset.
# These 10 products (5 real, 5 fictional) get stimuli from Claude + GPT-4 + human.
# Every product x register x generator = fully crossed on this subset.
CROSS_GENERATOR_SUBSET_IDS: list[str] = [
    "oral_care_real_01",
    "pet_food_real_01",
    "home_cleaning_real_01",
    "sports_nutrition_real_01",
    "skincare_real_01",
    "oral_care_fict_01",
    "pet_food_fict_01",
    "home_cleaning_fict_01",
    "sports_nutrition_fict_01",
    "skincare_fict_01",
]
# This yields 10 products x 5 registers x 3 generators = 150 additional stimuli
# for the cross-generator analysis.
```

**Acceptance criteria for Step 0.2:**
- All 40 real products defined with 3-5 core attributes each
- All 40 fictional products defined with 3-5 core attributes each
- Fictional products have plausible but non-existent names (verify with web search)
- No two products in the same category share more than 1 core attribute
- 10 products selected for cross-generator subset (5 real, 5 fictional, spanning 5 categories)

### Step 0.3: Create Debug Config

**File: `config/debug.yaml`**

```yaml
experiment:
  name: "plh-debug"
  seed: 42
  output_dir: "data/debug"

stimuli:
  n_categories: 2
  n_products_per_category: 2
  n_fictional_per_category: 1
  registers: ["marketing", "social"]
  variants_per_register: 1
  generators:
    primary: "anthropic"
    cross_validation_subset_size: 0
    cross_validation_generators: []

models:
  primary:
    name: "Qwen/Qwen2.5-1.5B"  # Tiny model for debug
    quantization: null
    device_map: "auto"
    batch_size: 2
  fp16_subset:
    name: "Qwen/Qwen2.5-1.5B"
    quantization: null
    device_map: "auto"
    batch_size: 1
    stimulus_subset_fraction: 1.0
  validation:
    name: "Qwen/Qwen2.5-1.5B"
    quantization: null
    device_map: "auto"
    batch_size: 2

extraction:
  components: ["hidden_states"]
  pooling: ["mean_no_special"]
  save_format: "hdf5"
  checkpoint_every: 5

probes:
  tasks:
    - name: "product_4class"
      n_classes: 4
      primary: true
    - name: "category_2class"
      n_classes: 2
    - name: "register_2class"
      n_classes: 2
  regularization: 1.0
  cv_folds: 2
  cv_stratify_by: ["category"]
  metrics: ["macro_f1"]
  bootstrap_ci: 0.95
  bootstrap_n: 100
```

**Verification**: Run `python -c "from plh.config import load_config; c = load_config('config/debug.yaml'); print(c.model_dump_json(indent=2))"` -- must parse without errors.

---

## 4. Stage 1: Stimulus Generation

### Design Decisions

- **Total stimuli**: 80 products x 5 registers x 2 variants = 800 primary stimuli
- **Cross-generator subset**: 10 products x 5 registers x 2 additional generators x 2 variants = 200 extra stimuli
- **Grand total**: ~1000 stimuli (800 primary + 200 cross-generator)
- **Generator**: Anthropic Claude (claude-sonnet-4-20250514) as primary; OpenAI GPT-4 (gpt-4o) for cross-generator subset
- **Human stimuli**: For the cross-generator subset, the "human" condition is a stretch goal; implementation should support it but scripts should work without it

### Step 1.1: Stimulus Schema

**File: `src/plh/stage1_stimuli/schema.py`**

```python
"""Pydantic models for stimulus data."""

from pydantic import BaseModel, field_validator
from typing import Optional


class Stimulus(BaseModel):
    """A single product description in a specific register."""
    stimulus_id: str          # "{product_id}_{register}_{variant}_{generator}"
    product_id: str           # References Product.id in constants.py
    product_name: str
    category: str             # Category enum value
    register: str             # Register enum value
    variant: int              # 0 or 1 (two variants per register)
    text: str                 # The actual description
    token_count: int          # Tokens as counted by the TARGET model's tokenizer
    core_attributes_present: list[str]  # Which core attrs are expressed in this text
    is_fictional: bool
    generator: str            # "anthropic", "openai", or "human"
    generation_model: Optional[str] = None  # e.g., "claude-sonnet-4-20250514"
    generation_timestamp: Optional[str] = None

    @field_validator("token_count")
    @classmethod
    def token_count_in_range(cls, v):
        if v < 40 or v > 250:
            raise ValueError(f"Token count {v} far outside target range [80, 150]")
        return v


class StimulusDataset(BaseModel):
    """Complete stimulus dataset."""
    version: str = "1.0"
    generation_date: str
    config_hash: str          # SHA256 of the config used to generate
    stimuli: list[Stimulus]
    metadata: dict            # Summary stats

    def get_by_product(self, product_id: str) -> list[Stimulus]:
        return [s for s in self.stimuli if s.product_id == product_id]

    def get_by_register(self, register: str) -> list[Stimulus]:
        return [s for s in self.stimuli if s.register == register]

    def get_by_generator(self, generator: str) -> list[Stimulus]:
        return [s for s in self.stimuli if s.generator == generator]
```

### Step 1.2: Prompt Templates

**File: `src/plh/stage1_stimuli/prompts.py`**

```python
"""Prompt templates for stimulus generation.

Each function returns a prompt that instructs the LLM to generate
a product description in a specific register, grounded in the
product's core attributes.
"""

from plh.constants import Product, Register, REGISTER_SPECS


def build_generation_prompt(
    product: Product,
    register: Register,
    variant_index: int,
) -> str:
    """Build a single stimulus generation prompt.

    Args:
        product: Product definition with core attributes
        register: Target register
        variant_index: 0 or 1 -- for variant diversity instruction

    Returns:
        System + user prompt as a single string for the LLM.
    """
    spec = REGISTER_SPECS[register]
    attrs_block = "\n".join(f"  - {a}" for a in product.core_attributes)
    features_block = "\n".join(f"  - {f}" for f in product.distinguishing_features)

    variant_instruction = (
        "Write a FIRST version of this description."
        if variant_index == 0
        else "Write a SECOND, distinctly different version of this description. "
             "Use different sentence structures, vocabulary choices, and opening hooks "
             "than you would in a first attempt, while conveying the same core facts."
    )

    return f"""You are generating a product description for a research experiment studying text register variation.

PRODUCT: {product.name}
CATEGORY: {product.category.value}
IS FICTIONAL: {product.is_fictional}

CORE FACTUAL ATTRIBUTES (ALL must be expressed in the description):
{attrs_block}

DISTINGUISHING FEATURES (include at least one):
{features_block}

TARGET REGISTER: {register.value}
Register specifications:
  - Voice: {spec['voice']}
  - Tone: {spec['tone']}
  - Structure: {spec['structure']}
  - Vocabulary style: {spec['vocabulary']}
  - Resembles: {spec['example_source']}

CONSTRAINTS:
  - Length: 80-150 tokens (STRICT -- count carefully)
  - Every core factual attribute MUST be conveyed (rephrased for register, not verbatim)
  - Do NOT include the product name as a heading or label
  - Do NOT include meta-commentary about the register
  - Write ONLY the product description, nothing else

{variant_instruction}
"""


def build_batch_generation_prompt(
    product: Product,
    registers: list[Register],
) -> str:
    """Build a prompt that generates ALL register variants for one product at once.

    This is more token-efficient but may produce less diverse variants.
    Used as a fallback if single-prompt generation is too slow.

    Returns a prompt expecting a JSON response with register keys.
    """
    attrs_block = "\n".join(f"  - {a}" for a in product.core_attributes)
    features_block = "\n".join(f"  - {f}" for f in product.distinguishing_features)

    register_specs_block = ""
    for reg in registers:
        spec = REGISTER_SPECS[reg]
        register_specs_block += f"""
  {reg.value}:
    Voice: {spec['voice']}
    Tone: {spec['tone']}
    Structure: {spec['structure']}
    Vocabulary: {spec['vocabulary']}
    Resembles: {spec['example_source']}
"""

    return f"""You are generating product descriptions in multiple text registers for a research experiment.

PRODUCT: {product.name}
CATEGORY: {product.category.value}
IS FICTIONAL: {product.is_fictional}

CORE FACTUAL ATTRIBUTES (ALL must appear in EVERY description):
{attrs_block}

DISTINGUISHING FEATURES (include at least one in each):
{features_block}

For EACH of the following registers, write TWO distinct variants (variant_0 and variant_1).
Each variant must be 80-150 tokens and convey all core attributes.
The two variants should differ in sentence structure, vocabulary, and opening hooks.

REGISTERS:
{register_specs_block}

Respond in JSON format:
{{
  "{registers[0].value}": {{
    "variant_0": "...",
    "variant_1": "..."
  }},
  ...
}}

Write ONLY the JSON, no commentary.
"""
```

### Step 1.3: Generation Orchestrator

**File: `src/plh/stage1_stimuli/generate.py`**

```python
"""Stimulus generation orchestrator.

Generates all stimuli using configured LLM providers,
validates semantic anchoring, and saves to disk.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from anthropic import Anthropic
from openai import OpenAI
from tqdm import tqdm

from plh.config import Config
from plh.constants import (
    REAL_PRODUCTS, FICTIONAL_PRODUCTS, Product, Register,
    CROSS_GENERATOR_SUBSET_IDS,
)
from plh.stage1_stimuli.prompts import build_generation_prompt
from plh.stage1_stimuli.schema import Stimulus, StimulusDataset
from plh.stage1_stimuli.validate import validate_stimulus, count_tokens
from plh.utils.checkpoint import StageCheckpoint


def generate_all_stimuli(config: Config) -> StimulusDataset:
    """Main entry point: generate all stimuli per config.

    Pipeline:
    1. Select products based on config (n_categories, n_per_category)
    2. For each product x register x variant: generate via primary generator
    3. For cross-generator subset: also generate via secondary generators
    4. Validate all stimuli (token count, core attribute coverage)
    5. Retry failures up to 3 times with adjusted prompts
    6. Save dataset with metadata

    Returns:
        StimulusDataset with all generated stimuli
    """
    # Implementation outline:
    products = _select_products(config)
    checkpoint = StageCheckpoint("stage1", config.experiment.output_dir)
    stimuli: list[Stimulus] = checkpoint.load_partial() or []
    completed_ids = {s.stimulus_id for s in stimuli}

    # Primary generation (Claude)
    anthropic_client = Anthropic()  # Uses ANTHROPIC_API_KEY from env
    for product in tqdm(products, desc="Generating stimuli"):
        for register in Register:
            for variant in range(config.stimuli.variants_per_register):
                stim_id = f"{product.id}_{register.value}_{variant}_anthropic"
                if stim_id in completed_ids:
                    continue
                stimulus = _generate_single(
                    product, register, variant, "anthropic",
                    anthropic_client=anthropic_client, config=config,
                )
                stimuli.append(stimulus)
                if len(stimuli) % 20 == 0:
                    checkpoint.save_partial(stimuli)

    # Cross-generator subset (GPT-4)
    if config.stimuli.generators.get("cross_validation_subset_size", 0) > 0:
        openai_client = OpenAI()  # Uses OPENAI_API_KEY from env
        subset_products = [p for p in products if p.id in CROSS_GENERATOR_SUBSET_IDS]
        for product in tqdm(subset_products, desc="Cross-generator (GPT-4)"):
            for register in Register:
                for variant in range(config.stimuli.variants_per_register):
                    stim_id = f"{product.id}_{register.value}_{variant}_openai"
                    if stim_id in completed_ids:
                        continue
                    stimulus = _generate_single(
                        product, register, variant, "openai",
                        openai_client=openai_client, config=config,
                    )
                    stimuli.append(stimulus)

    # Validate all
    failures = []
    for stim in stimuli:
        issues = validate_stimulus(stim, _get_product_by_id(stim.product_id, products))
        if issues:
            failures.append((stim.stimulus_id, issues))

    # Retry failures (up to 3 attempts)
    # ... (retry logic with stricter prompts)

    dataset = StimulusDataset(
        version="1.0",
        generation_date=datetime.now(timezone.utc).isoformat(),
        config_hash=hashlib.sha256(json.dumps(config.model_dump(), sort_keys=True).encode()).hexdigest(),
        stimuli=stimuli,
        metadata={
            "n_stimuli": len(stimuli),
            "n_real_products": sum(1 for p in products if not p.is_fictional),
            "n_fictional_products": sum(1 for p in products if p.is_fictional),
            "n_registers": len(Register),
            "n_variants_per_register": config.stimuli.variants_per_register,
            "n_validation_failures": len(failures),
            "generators": ["anthropic", "openai"],
        },
    )
    return dataset


def _generate_single(
    product: Product,
    register: Register,
    variant: int,
    generator: str,
    config: Config,
    anthropic_client: Optional[Anthropic] = None,
    openai_client: Optional[OpenAI] = None,
) -> Stimulus:
    """Generate a single stimulus via the specified generator.

    Calls the appropriate API, parses the response, counts tokens,
    and constructs a Stimulus object.
    """
    prompt = build_generation_prompt(product, register, variant)

    if generator == "anthropic":
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        gen_model = "claude-sonnet-4-20250514"
    elif generator == "openai":
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            max_tokens=300,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content.strip()
        gen_model = "gpt-4o"
    else:
        raise ValueError(f"Unknown generator: {generator}")

    token_count = count_tokens(text)  # Uses target model tokenizer

    return Stimulus(
        stimulus_id=f"{product.id}_{register.value}_{variant}_{generator}",
        product_id=product.id,
        product_name=product.name,
        category=product.category.value,
        register=register.value,
        variant=variant,
        text=text,
        token_count=token_count,
        core_attributes_present=product.core_attributes,  # Validated later
        is_fictional=product.is_fictional,
        generator=generator,
        generation_model=gen_model,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _select_products(config: Config) -> list[Product]:
    """Select products per config (may use subset for debug)."""
    categories = list(Category)[:config.stimuli.n_categories]
    selected = []
    for cat in categories:
        reals = [p for p in REAL_PRODUCTS if p.category == cat]
        ficts = [p for p in FICTIONAL_PRODUCTS if p.category == cat]
        selected.extend(reals[:config.stimuli.n_products_per_category])
        selected.extend(ficts[:config.stimuli.n_fictional_per_category])
    return selected


def _get_product_by_id(product_id: str, products: list[Product]) -> Product:
    for p in products:
        if p.id == product_id:
            return p
    raise ValueError(f"Product {product_id} not found")
```

### Step 1.4: Semantic Validation

**File: `src/plh/stage1_stimuli/validate.py`**

```python
"""Stimulus validation: token counts, semantic anchoring, length compliance."""

from transformers import AutoTokenizer
from plh.constants import Product
from plh.stage1_stimuli.schema import Stimulus

# Lazy-loaded tokenizer (uses the primary model's tokenizer)
_tokenizer = None

def get_tokenizer(model_name: str = "Qwen/Qwen2.5-27B") -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer


def count_tokens(text: str, model_name: str = "Qwen/Qwen2.5-27B") -> int:
    """Count tokens using the target model's tokenizer."""
    tokenizer = get_tokenizer(model_name)
    return len(tokenizer.encode(text))


def validate_stimulus(stimulus: Stimulus, product: Product) -> list[str]:
    """Validate a single stimulus against its product definition.

    Returns list of issues (empty = valid).

    Checks:
    1. Token count within target range (warn if 60-80 or 150-180, fail if outside)
    2. Core attributes mentioned (fuzzy string matching)
    3. No register-breaking artifacts (e.g., markdown headers in social post)
    """
    issues = []

    # Token count
    if stimulus.token_count < 60 or stimulus.token_count > 200:
        issues.append(f"Token count {stimulus.token_count} outside acceptable range [60, 200]")
    elif stimulus.token_count < 80 or stimulus.token_count > 150:
        issues.append(f"Token count {stimulus.token_count} outside target range [80, 150] (warning)")

    # Core attribute coverage (simple keyword matching -- not perfect but catches gross omissions)
    for attr in product.core_attributes:
        # Extract key numbers/terms from attribute
        key_terms = _extract_key_terms(attr)
        text_lower = stimulus.text.lower()
        if not any(term.lower() in text_lower for term in key_terms):
            issues.append(f"Core attribute likely missing: '{attr}' (no key terms found)")

    return issues


def _extract_key_terms(attribute: str) -> list[str]:
    """Extract key searchable terms from a core attribute string.

    E.g., '40,000 brush strokes per minute' -> ['40,000', '40000', 'brush strokes']
    """
    import re
    terms = []
    # Numbers (with and without commas)
    numbers = re.findall(r'\d[\d,.]*\d|\d', attribute)
    for n in numbers:
        terms.append(n)
        terms.append(n.replace(",", ""))
    # Multi-word technical phrases (3+ chars, not stop words)
    stop_words = {'the', 'and', 'with', 'for', 'per', 'that', 'from', 'into'}
    words = [w for w in attribute.lower().split() if len(w) > 3 and w not in stop_words]
    if len(words) >= 2:
        terms.append(" ".join(words[:3]))
    return terms if terms else [attribute[:20].lower()]
```

### Step 1.5: Entry Point Script

**File: `scripts/run_stage1.py`**

```python
"""Stage 1: Stimulus Generation.

Usage:
    python scripts/run_stage1.py --config config/default.yaml
    python scripts/run_stage1.py --config config/debug.yaml
"""

import argparse
import json
from pathlib import Path

from plh.config import load_config
from plh.stage1_stimuli.generate import generate_all_stimuli
from plh.utils.seeds import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Generate experiment stimuli")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", default=None, help="Override output path")
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config.experiment.seed)

    dataset = generate_all_stimuli(config)

    output_dir = Path(args.output or config.experiment.output_dir) / "stimuli"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stimuli.json"

    with open(output_path, "w") as f:
        f.write(dataset.model_dump_json(indent=2))

    print(f"Generated {len(dataset.stimuli)} stimuli")
    print(f"Saved to {output_path}")
    print(f"Validation failures in metadata: {dataset.metadata.get('n_validation_failures', 0)}")


if __name__ == "__main__":
    main()
```

### Stage 1 Acceptance Criteria

1. `stimuli.json` contains 800+ stimuli (80 products x 5 registers x 2 variants)
2. Cross-generator subset contains 200 additional stimuli (10 products x 5 registers x 2 variants x 2 extra generators)
3. Every stimulus has `token_count` in [60, 200] (target [80, 150])
4. Every stimulus passes `validate_stimulus` with at most warnings (no hard failures)
5. `core_attributes_present` field correctly reflects which attributes appear
6. Fictional product stimuli do not reference real brand names
7. All stimuli parseable by the target model's tokenizer

### Stage 1 Verification Command

```bash
python scripts/run_stage1.py --config config/debug.yaml
python -c "
import json
with open('data/debug/stimuli/stimuli.json') as f:
    d = json.load(f)
print(f'Total stimuli: {len(d[\"stimuli\"])}')
cats = set(s['category'] for s in d['stimuli'])
regs = set(s['register'] for s in d['stimuli'])
print(f'Categories: {cats}')
print(f'Registers: {regs}')
"
```

---

## 5. Stage 2: Hidden State Extraction

### Memory Budget Analysis

**Qwen3.5-27B (4-bit GPTQ):**
- Model: ~14 GB VRAM
- Per-stimulus peak: ~1-2 GB (forward pass with hidden states)
- Available for activations: ~16 GB
- Batch size 4 should be safe; start with 1, increase until VRAM limit

**Qwen3.5-27B (FP16, CPU offloading):**
- Model: ~54 GB (split: ~30 GB GPU, ~24 GB CPU RAM)
- Must use batch_size=1
- Forward pass will be slow (~10-30s per stimulus)
- Run on 25% subset (200 stimuli) = ~1-2 hours

**Llama-3.1-8B-Instruct (FP16):**
- Model: ~16 GB
- Available for activations: ~16 GB
- Batch size 8 feasible

**Per-model output size (800 stimuli, ~64 layers, hidden_dim=3584 for Qwen):**
- hidden_states: 800 x 64 x 3584 x 4 bytes = ~700 MB per pooling strategy
- attention_output: same = ~700 MB
- mlp_output: same = ~700 MB
- residual_stream: same = ~700 MB
- Total per model: ~2.8 GB per pooling strategy, ~5.6 GB with both pooling strategies
- Grand total (3 models): ~17 GB

### Step 2.1: Model Loading

**File: `src/plh/stage2_extraction/models.py`**

```python
"""Model loading utilities for hidden state extraction.

Supports:
- GPTQ/AWQ 4-bit quantized models
- FP16 models with automatic CPU offloading
- Full-precision models that fit in VRAM
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from plh.config import ModelSpec


def load_model_and_tokenizer(
    spec: ModelSpec,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model per spec. Returns (model, tokenizer).

    The model is loaded in eval mode with no gradient computation.
    For quantized models, uses auto-gptq or autoawq backend.
    For FP16 with device_map="auto", layers are split across GPU and CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(spec.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": spec.device_map,
        "output_hidden_states": True,  # CRITICAL: must be set at load time for some models
    }

    if spec.quantization == "gptq-4bit":
        load_kwargs["torch_dtype"] = torch.float16
        # auto-gptq models are loaded via transformers' GPTQ integration
        load_kwargs["quantization_config"] = None  # Uses model's built-in GPTQ config
        # NOTE: Use the GPTQ variant of the model from HuggingFace
        # e.g., "Qwen/Qwen2.5-27B-GPTQ-Int4" or similar
    elif spec.quantization is None:
        load_kwargs["torch_dtype"] = torch.float16
    else:
        raise ValueError(f"Unknown quantization: {spec.quantization}")

    model = AutoModelForCausalLM.from_pretrained(spec.name, **load_kwargs)
    model.eval()

    return model, tokenizer


def get_layer_count(model: AutoModelForCausalLM) -> int:
    """Get the number of transformer layers in the model."""
    if hasattr(model.config, "num_hidden_layers"):
        return model.config.num_hidden_layers
    raise AttributeError("Cannot determine layer count from model config")


def get_hidden_dim(model: AutoModelForCausalLM) -> int:
    """Get the hidden dimension of the model."""
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    raise AttributeError("Cannot determine hidden dimension from model config")
```

### Step 2.2: Forward Hooks for Component Decomposition

**File: `src/plh/stage2_extraction/hooks.py`**

```python
"""Forward hooks for extracting attention output, MLP output, and residual stream.

HuggingFace's output_hidden_states=True gives us the residual stream after
each layer. To decompose into attention and MLP contributions, we need hooks.

Architecture assumption (Qwen2/Llama-style):
    For each layer l:
        residual = input
        attn_out = self_attn(layer_norm(residual))
        residual = residual + attn_out              # post-attention residual
        mlp_out  = mlp(layer_norm(residual))
        residual = residual + mlp_out               # post-MLP residual (= hidden_state[l])
"""

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM
from dataclasses import dataclass, field


@dataclass
class LayerActivations:
    """Captured activations for a single layer."""
    attention_output: Tensor | None = None  # Shape: (batch, seq_len, hidden_dim)
    mlp_output: Tensor | None = None        # Shape: (batch, seq_len, hidden_dim)


@dataclass
class ExtractionHooks:
    """Manages forward hooks for activation extraction."""
    activations: dict[int, LayerActivations] = field(default_factory=dict)
    _handles: list = field(default_factory=list)

    def register(self, model: AutoModelForCausalLM) -> None:
        """Register forward hooks on all transformer layers.

        Must be called before the forward pass.
        Hooks capture attention output and MLP output at each layer.
        """
        self.activations.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

        layers = _get_transformer_layers(model)
        for layer_idx, layer in enumerate(layers):
            self.activations[layer_idx] = LayerActivations()

            # Hook on self_attn module output
            attn_module = _get_attn_module(layer)
            handle = attn_module.register_forward_hook(
                self._make_attn_hook(layer_idx)
            )
            self._handles.append(handle)

            # Hook on MLP module output
            mlp_module = _get_mlp_module(layer)
            handle = mlp_module.register_forward_hook(
                self._make_mlp_hook(layer_idx)
            )
            self._handles.append(handle)

    def _make_attn_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output is typically a tuple; first element is the attention output tensor
            if isinstance(output, tuple):
                self.activations[layer_idx].attention_output = output[0].detach().cpu()
            else:
                self.activations[layer_idx].attention_output = output.detach().cpu()
        return hook

    def _make_mlp_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.activations[layer_idx].mlp_output = output[0].detach().cpu()
            else:
                self.activations[layer_idx].mlp_output = output.detach().cpu()
        return hook

    def remove(self) -> None:
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def get_all(self) -> dict[int, LayerActivations]:
        """Return captured activations (copy to avoid mutation)."""
        return dict(self.activations)


def _get_transformer_layers(model):
    """Navigate model architecture to find the list of transformer layers.

    Supports Qwen2 and Llama architectures.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers  # Llama, Qwen2
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2 style
    raise AttributeError(
        f"Cannot find transformer layers in model of type {type(model).__name__}. "
        "Inspect model architecture and update _get_transformer_layers()."
    )


def _get_attn_module(layer):
    """Get the self-attention module from a transformer layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn  # Llama, Qwen2
    if hasattr(layer, "attn"):
        return layer.attn  # GPT-2 style
    raise AttributeError(f"Cannot find attention module in layer {type(layer).__name__}")


def _get_mlp_module(layer):
    """Get the MLP module from a transformer layer."""
    if hasattr(layer, "mlp"):
        return layer.mlp
    if hasattr(layer, "feed_forward"):
        return layer.feed_forward
    raise AttributeError(f"Cannot find MLP module in layer {type(layer).__name__}")
```

### Step 2.3: Pooling Strategies

**File: `src/plh/stage2_extraction/pooling.py`**

```python
"""Pooling strategies for converting sequence-level hidden states to fixed-size vectors."""

import torch
from torch import Tensor


def mean_pool_no_special(
    hidden_states: Tensor,
    attention_mask: Tensor,
    tokenizer_special_ids: set[int] | None = None,
    input_ids: Tensor | None = None,
) -> Tensor:
    """Mean pool hidden states, excluding special tokens (BOS, EOS, PAD).

    Args:
        hidden_states: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len) -- 1 for real tokens, 0 for padding
        tokenizer_special_ids: set of token IDs to exclude (BOS, EOS, etc.)
        input_ids: (batch, seq_len) -- needed if excluding specific special token IDs

    Returns:
        (batch, hidden_dim) -- mean-pooled representation
    """
    mask = attention_mask.clone().float()

    # Zero out special tokens if IDs provided
    if tokenizer_special_ids and input_ids is not None:
        for special_id in tokenizer_special_ids:
            mask[input_ids == special_id] = 0.0

    # Expand mask for broadcasting: (batch, seq_len) -> (batch, seq_len, 1)
    mask_expanded = mask.unsqueeze(-1)

    # Masked mean
    summed = (hidden_states * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1e-9)

    return summed / counts


def last_token_pool(
    hidden_states: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """Extract the hidden state of the last non-padding token.

    Args:
        hidden_states: (batch, seq_len, hidden_dim)
        attention_mask: (batch, seq_len)

    Returns:
        (batch, hidden_dim) -- last-token representation
    """
    # Find the index of the last 1 in attention_mask for each batch item
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, seq_lengths]
```

### Step 2.4: Extraction Pipeline

**File: `src/plh/stage2_extraction/extract.py`**

```python
"""Hidden state extraction pipeline.

For each model x stimulus: forward pass -> hook capture -> pool -> save to HDF5.
Supports checkpointing (resume after crash).
"""

import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

from plh.config import Config, ModelSpec
from plh.stage1_stimuli.schema import StimulusDataset, Stimulus
from plh.stage2_extraction.models import load_model_and_tokenizer, get_layer_count, get_hidden_dim
from plh.stage2_extraction.hooks import ExtractionHooks
from plh.stage2_extraction.pooling import mean_pool_no_special, last_token_pool
from plh.utils.checkpoint import StageCheckpoint


def extract_hidden_states(
    config: Config,
    stimuli: StimulusDataset,
    model_key: str,  # "primary", "fp16_subset", or "validation"
) -> Path:
    """Extract hidden states for all stimuli using the specified model.

    Args:
        config: Experiment configuration
        stimuli: Loaded stimulus dataset
        model_key: Which model config to use

    Returns:
        Path to the output HDF5 file

    HDF5 file structure:
        /stimulus_ids          : string dataset of stimulus IDs (N,)
        /metadata/model_name   : str
        /metadata/n_layers     : int
        /metadata/hidden_dim   : int
        /metadata/n_stimuli    : int
        /hidden_states/mean    : float32 (N, L+1, D)  -- L+1 because layer 0 = embedding
        /hidden_states/last    : float32 (N, L+1, D)
        /attention_output/mean : float32 (N, L, D)
        /attention_output/last : float32 (N, L, D)
        /mlp_output/mean       : float32 (N, L, D)
        /mlp_output/last       : float32 (N, L, D)
    """
    model_spec: ModelSpec = getattr(config.models, model_key)

    # Subset selection for fp16 runs
    stim_list = stimuli.stimuli
    if model_spec.stimulus_subset_fraction and model_spec.stimulus_subset_fraction < 1.0:
        n_subset = int(len(stim_list) * model_spec.stimulus_subset_fraction)
        # Deterministic subset: every Nth stimulus
        step = max(1, len(stim_list) // n_subset)
        stim_list = stim_list[::step][:n_subset]

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_spec)
    n_layers = get_layer_count(model)
    hidden_dim = get_hidden_dim(model)

    special_ids = set()
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        special_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        special_ids.add(tokenizer.pad_token_id)

    # Setup hooks for component decomposition
    hooks = ExtractionHooks()
    if "attention_output" in config.extraction.components or "mlp_output" in config.extraction.components:
        hooks.register(model)

    # Output file
    output_dir = Path(config.experiment.output_dir) / "hidden_states"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_key}_states.h5"

    # Determine which stimuli are already extracted (for resume)
    completed_ids = set()
    if output_path.exists():
        with h5py.File(output_path, "r") as f:
            if "stimulus_ids" in f:
                completed_ids = set(f["stimulus_ids"].asstr()[:])

    remaining = [s for s in stim_list if s.stimulus_id not in completed_ids]
    print(f"Model: {model_spec.name} ({model_key})")
    print(f"Total stimuli: {len(stim_list)}, already done: {len(completed_ids)}, remaining: {len(remaining)}")

    # Process in batches
    batch_size = model_spec.batch_size
    all_results = []  # List of dicts with pooled tensors

    for batch_start in tqdm(range(0, len(remaining), batch_size), desc=f"Extracting ({model_key})"):
        batch_stimuli = remaining[batch_start:batch_start + batch_size]
        texts = [s.text for s in batch_stimuli]

        # Tokenize
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        # Move to model's device (first parameter's device)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (n_layers+1) tensors, each (batch, seq, dim)
        # Layer 0 = embedding output, layers 1..N = transformer layer outputs

        for i, stim in enumerate(batch_stimuli):
            result = {"stimulus_id": stim.stimulus_id}

            # Hidden states (residual stream at each layer)
            if "hidden_states" in config.extraction.components:
                hs_stack = torch.stack([h[i:i+1] for h in outputs.hidden_states], dim=1)
                # hs_stack shape: (1, n_layers+1, seq_len, hidden_dim)
                # Squeeze batch dim, pool over seq_len

                for pool_name in config.extraction.pooling:
                    pooled_layers = []
                    for layer_idx in range(hs_stack.size(1)):
                        layer_hs = hs_stack[0, layer_idx:layer_idx+1]  # (1, seq_len, dim)
                        mask = inputs["attention_mask"][i:i+1]
                        if pool_name == "mean_no_special":
                            pooled = mean_pool_no_special(
                                layer_hs, mask, special_ids, inputs["input_ids"][i:i+1]
                            )
                        elif pool_name == "last_token":
                            pooled = last_token_pool(layer_hs, mask)
                        else:
                            raise ValueError(f"Unknown pooling: {pool_name}")
                        pooled_layers.append(pooled.cpu().numpy())

                    result[f"hidden_states_{pool_name}"] = np.stack(pooled_layers, axis=0).squeeze()
                    # Shape: (n_layers+1, hidden_dim)

            # Attention and MLP outputs from hooks
            if "attention_output" in config.extraction.components:
                hook_acts = hooks.get_all()
                for pool_name in config.extraction.pooling:
                    pooled_layers = []
                    for layer_idx in sorted(hook_acts.keys()):
                        attn_out = hook_acts[layer_idx].attention_output[i:i+1]  # (1, seq, dim)
                        mask = inputs["attention_mask"][i:i+1].cpu()
                        if pool_name == "mean_no_special":
                            pooled = mean_pool_no_special(
                                attn_out, mask, special_ids,
                                inputs["input_ids"][i:i+1].cpu()
                            )
                        elif pool_name == "last_token":
                            pooled = last_token_pool(attn_out, mask)
                        pooled_layers.append(pooled.numpy())
                    result[f"attention_output_{pool_name}"] = np.stack(pooled_layers, axis=0).squeeze()

            if "mlp_output" in config.extraction.components:
                hook_acts = hooks.get_all()
                for pool_name in config.extraction.pooling:
                    pooled_layers = []
                    for layer_idx in sorted(hook_acts.keys()):
                        mlp_out = hook_acts[layer_idx].mlp_output[i:i+1]
                        mask = inputs["attention_mask"][i:i+1].cpu()
                        if pool_name == "mean_no_special":
                            pooled = mean_pool_no_special(
                                mlp_out, mask, special_ids,
                                inputs["input_ids"][i:i+1].cpu()
                            )
                        elif pool_name == "last_token":
                            pooled = last_token_pool(mlp_out, mask)
                        pooled_layers.append(pooled.numpy())
                    result[f"mlp_output_{pool_name}"] = np.stack(pooled_layers, axis=0).squeeze()

            all_results.append(result)

        # Checkpoint periodically
        if len(all_results) % config.extraction.checkpoint_every == 0:
            _save_results_to_hdf5(output_path, all_results, n_layers, hidden_dim)
            all_results.clear()

        # Free GPU memory
        del outputs
        torch.cuda.empty_cache()

    # Final save
    if all_results:
        _save_results_to_hdf5(output_path, all_results, n_layers, hidden_dim)

    hooks.remove()

    print(f"Extraction complete. Saved to {output_path}")
    return output_path


def _save_results_to_hdf5(
    path: Path,
    results: list[dict],
    n_layers: int,
    hidden_dim: int,
) -> None:
    """Append extraction results to HDF5 file.

    Uses resizable datasets so results can be appended incrementally.
    """
    mode = "a" if path.exists() else "w"
    with h5py.File(path, mode) as f:
        # Stimulus IDs
        new_ids = [r["stimulus_id"] for r in results]
        if "stimulus_ids" in f:
            existing = list(f["stimulus_ids"].asstr()[:])
            existing.extend(new_ids)
            del f["stimulus_ids"]
            dt = h5py.string_dtype()
            f.create_dataset("stimulus_ids", data=existing, dtype=dt)
        else:
            dt = h5py.string_dtype()
            f.create_dataset("stimulus_ids", data=new_ids, dtype=dt,
                             maxshape=(None,))

        # Numeric datasets
        for key in results[0]:
            if key == "stimulus_id":
                continue
            new_data = np.stack([r[key] for r in results], axis=0)  # (batch, layers, dim)

            dataset_name = key  # e.g., "hidden_states_mean_no_special"
            if dataset_name in f:
                dset = f[dataset_name]
                old_size = dset.shape[0]
                dset.resize(old_size + new_data.shape[0], axis=0)
                dset[old_size:] = new_data
            else:
                maxshape = (None,) + new_data.shape[1:]
                f.create_dataset(
                    dataset_name, data=new_data,
                    dtype="float32", maxshape=maxshape,
                    chunks=True, compression="gzip", compression_opts=4,
                )
```

### Step 2.5: Entry Point

**File: `scripts/run_stage2.py`**

```python
"""Stage 2: Hidden State Extraction.

Usage:
    python scripts/run_stage2.py --config config/default.yaml --model primary
    python scripts/run_stage2.py --config config/default.yaml --model validation
    python scripts/run_stage2.py --config config/default.yaml --model fp16_subset
"""

import argparse
import json
from pathlib import Path

from plh.config import load_config
from plh.stage1_stimuli.schema import StimulusDataset
from plh.stage2_extraction.extract import extract_hidden_states
from plh.utils.seeds import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--model", required=True, choices=["primary", "fp16_subset", "validation"],
                        help="Which model to use")
    parser.add_argument("--stimuli", default=None, help="Override stimuli.json path")
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config.experiment.seed)

    # Load stimuli
    stimuli_path = args.stimuli or Path(config.experiment.output_dir) / "stimuli" / "stimuli.json"
    with open(stimuli_path) as f:
        stimuli = StimulusDataset(**json.load(f))

    print(f"Loaded {len(stimuli.stimuli)} stimuli")

    output_path = extract_hidden_states(config, stimuli, args.model)
    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
```

### Stage 2 Acceptance Criteria

1. HDF5 file for each model contains all expected stimuli
2. Hidden state shapes are correct: (N, L+1, D) for hidden_states, (N, L, D) for attention/MLP
3. No NaN or Inf values in any tensor
4. Checkpoint/resume works: killing and restarting produces same final output
5. FP16 subset contains the correct deterministic subset of stimuli
6. GPU memory stays under 30 GB during extraction (2 GB headroom)
7. Hooks correctly capture attention and MLP outputs (verify against manual decomposition for 1 stimulus)

### Stage 2 Verification

```bash
python scripts/run_stage2.py --config config/debug.yaml --model primary
python -c "
import h5py
f = h5py.File('data/debug/hidden_states/primary_states.h5', 'r')
print('Keys:', list(f.keys()))
print('Stimulus count:', len(f['stimulus_ids']))
for key in f:
    if key != 'stimulus_ids':
        print(f'{key}: shape={f[key].shape}, dtype={f[key].dtype}')
        import numpy as np
        print(f'  NaN count: {np.isnan(f[key][:]).sum()}')
        print(f'  Inf count: {np.isinf(f[key][:]).sum()}')
f.close()
"
```

---

## 6. Stage 3: Similarity Analysis

### Step 3.1: Representational Dissimilarity Matrices

**File: `src/plh/stage3_similarity/rdm.py`**

```python
"""Compute Representational Dissimilarity Matrices (RDMs) at each layer.

An RDM is a symmetric matrix where entry (i,j) = dissimilarity between
stimuli i and j. We use 1 - cosine_similarity as the dissimilarity metric.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, squareform


def compute_rdm(
    representations: NDArray,  # (N, hidden_dim)
    metric: str = "cosine",
) -> NDArray:
    """Compute the RDM for a set of representations.

    Args:
        representations: (N, D) array of stimulus representations at one layer
        metric: distance metric ('cosine', 'correlation', 'euclidean')

    Returns:
        (N, N) symmetric dissimilarity matrix
    """
    # pdist returns condensed distance matrix; squareform makes it (N, N)
    distances = pdist(representations, metric=metric)
    return squareform(distances)


def compute_rdms_all_layers(
    hidden_states: NDArray,  # (N, L, D)
    metric: str = "cosine",
) -> NDArray:
    """Compute RDMs at every layer.

    Args:
        hidden_states: (N, L, D) array
        metric: distance metric

    Returns:
        (L, N, N) array of RDMs
    """
    n_stimuli, n_layers, hidden_dim = hidden_states.shape
    rdms = np.zeros((n_layers, n_stimuli, n_stimuli), dtype=np.float32)
    for layer in range(n_layers):
        rdms[layer] = compute_rdm(hidden_states[:, layer, :], metric)
    return rdms


def build_model_rdm(
    stimulus_ids: list[str],
    product_ids: list[str],
    category_ids: list[str],
    same_product: float = 0.0,
    same_category: float = 0.5,
    different_category: float = 1.0,
) -> NDArray:
    """Build the theoretical model RDM reflecting the hypothesis.

    Args:
        stimulus_ids: list of stimulus IDs (length N)
        product_ids: list of product IDs for each stimulus
        category_ids: list of category IDs for each stimulus
        same_product: dissimilarity for same-product pairs
        same_category: dissimilarity for different-product, same-category pairs
        different_category: dissimilarity for different-category pairs

    Returns:
        (N, N) model RDM
    """
    n = len(stimulus_ids)
    rdm = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            if product_ids[i] == product_ids[j]:
                rdm[i, j] = rdm[j, i] = same_product
            elif category_ids[i] == category_ids[j]:
                rdm[i, j] = rdm[j, i] = same_category
            else:
                rdm[i, j] = rdm[j, i] = different_category
    return rdm
```

### Step 3.2: Anisotropy Correction

**File: `src/plh/stage3_similarity/anisotropy.py`**

```python
"""Anisotropy correction methods for hidden state representations.

Implements:
1. Mean centering: subtract the global mean direction
2. Whitening: mean centering + decorrelation + variance normalization
3. No correction (passthrough)
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


def correct_anisotropy(
    representations: NDArray,  # (N, D)
    method: str = "none",
    n_components: int | None = None,
) -> NDArray:
    """Apply anisotropy correction to a set of representations.

    Args:
        representations: (N, D) array
        method: "none", "mean_centering", or "whitening"
        n_components: for whitening, number of PCA components to keep.
                      If None, keep all. Recommend keeping components
                      explaining >99% variance to avoid noise amplification.

    Returns:
        (N, D') corrected representations (D' = D for centering, n_components for whitening)
    """
    if method == "none":
        return representations.copy()

    elif method == "mean_centering":
        mean = representations.mean(axis=0, keepdims=True)
        return representations - mean

    elif method == "whitening":
        # Step 1: mean center
        mean = representations.mean(axis=0, keepdims=True)
        centered = representations - mean

        # Step 2: PCA whitening
        if n_components is None:
            n_components = min(centered.shape[0] - 1, centered.shape[1])

        pca = PCA(n_components=n_components, whiten=True)
        whitened = pca.fit_transform(centered)

        # Log how much variance is retained
        var_explained = pca.explained_variance_ratio_.sum()
        if var_explained < 0.95:
            import warnings
            warnings.warn(
                f"Whitening with {n_components} components retains only "
                f"{var_explained:.1%} of variance. Consider increasing n_components."
            )

        return whitened

    else:
        raise ValueError(f"Unknown anisotropy correction method: {method}")


def correct_anisotropy_all_layers(
    hidden_states: NDArray,  # (N, L, D)
    method: str = "none",
    n_components: int | None = None,
) -> NDArray:
    """Apply anisotropy correction independently at each layer.

    Returns:
        (N, L, D') array with corrected representations
    """
    n_stimuli, n_layers, hidden_dim = hidden_states.shape
    results = []
    for layer in range(n_layers):
        corrected = correct_anisotropy(
            hidden_states[:, layer, :], method, n_components
        )
        results.append(corrected)

    # Stack -- note: whitened representations may have different D than original
    return np.stack(results, axis=1)  # (L, N, D') -> need to transpose
    # Actually: results is list of (N, D'), stack along axis=1 won't work if D' varies
    # Since we apply same n_components at each layer, D' is consistent
    # Correct approach:
    # return np.stack(results, axis=1)  # This creates (N, L, D') if results are (N, D')
    # Wait, np.stack on list of (N, D') along axis=1 gives (N, L, D'). Yes, correct.
```

### Step 3.3: RSA Computation

**File: `src/plh/stage3_similarity/rsa.py`**

```python
"""Representational Similarity Analysis (RSA).

Correlates observed RDMs with model (theoretical) RDMs to test
whether hidden-state geometry matches the predicted semantic structure.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import squareform


def rsa_correlation(
    observed_rdm: NDArray,  # (N, N) observed dissimilarity matrix
    model_rdm: NDArray,     # (N, N) theoretical dissimilarity matrix
    method: str = "spearman",
) -> tuple[float, float]:
    """Compute RSA correlation between observed and model RDMs.

    Only uses the upper triangle (excluding diagonal) to avoid
    double-counting and self-similarity.

    Args:
        observed_rdm: (N, N) observed RDM
        model_rdm: (N, N) model RDM
        method: "spearman" or "pearson"

    Returns:
        (correlation, p_value)
    """
    # Extract upper triangle (excluding diagonal)
    obs_vec = squareform(observed_rdm, checks=False)
    mod_vec = squareform(model_rdm, checks=False)

    if method == "spearman":
        return spearmanr(obs_vec, mod_vec)
    elif method == "pearson":
        return pearsonr(obs_vec, mod_vec)
    else:
        raise ValueError(f"Unknown RSA method: {method}")


def rsa_all_layers(
    observed_rdms: NDArray,  # (L, N, N)
    model_rdm: NDArray,      # (N, N)
    method: str = "spearman",
) -> tuple[NDArray, NDArray]:
    """Compute RSA correlation at every layer.

    Returns:
        (correlations, p_values) -- each shape (L,)
    """
    n_layers = observed_rdms.shape[0]
    correlations = np.zeros(n_layers)
    p_values = np.zeros(n_layers)

    for layer in range(n_layers):
        corr, pval = rsa_correlation(observed_rdms[layer], model_rdm, method)
        correlations[layer] = corr
        p_values[layer] = pval

    return correlations, p_values


def rsa_permutation_test(
    observed_rdm: NDArray,
    model_rdm: NDArray,
    n_permutations: int = 10000,
    method: str = "spearman",
    seed: int = 42,
) -> tuple[float, float, NDArray]:
    """Permutation test for RSA significance.

    Shuffles rows/columns of the model RDM to generate null distribution.

    Returns:
        (observed_correlation, p_value, null_distribution)
    """
    rng = np.random.RandomState(seed)
    obs_corr, _ = rsa_correlation(observed_rdm, model_rdm, method)

    n = model_rdm.shape[0]
    null_corrs = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm = rng.permutation(n)
        shuffled = model_rdm[perm][:, perm]
        null_corrs[i], _ = rsa_correlation(observed_rdm, shuffled, method)

    p_value = (np.sum(null_corrs >= obs_corr) + 1) / (n_permutations + 1)
    return obs_corr, p_value, null_corrs
```

### Step 3.4: Cosine Similarity Conditions

**File: `src/plh/stage3_similarity/cosine.py`**

```python
"""Per-layer cosine similarity matrices for the three experimental conditions.

Conditions:
- SP-DR: Same Product, Different Register
- DP-SC: Different Product, Same Category
- DC: Different Category
"""

import numpy as np
from numpy.typing import NDArray
from itertools import combinations


def compute_condition_similarities(
    hidden_states: NDArray,          # (N, L, D) or (N, D) for single layer
    product_ids: list[str],
    category_ids: list[str],
    register_ids: list[str],
) -> dict[str, NDArray]:
    """Compute mean cosine similarity for each condition at each layer.

    Args:
        hidden_states: (N, L, D) array of representations
        product_ids: product ID for each stimulus
        category_ids: category ID for each stimulus
        register_ids: register ID for each stimulus

    Returns:
        dict with keys "SP_DR", "DP_SC", "DC", each containing
        an array of shape (L,) with mean similarities per layer
    """
    if hidden_states.ndim == 2:
        hidden_states = hidden_states[:, np.newaxis, :]

    n_stimuli, n_layers, hidden_dim = hidden_states.shape

    # Classify all pairs
    sp_dr_pairs = []  # Same product, different register
    dp_sc_pairs = []  # Different product, same category
    dc_pairs = []     # Different category

    for i, j in combinations(range(n_stimuli), 2):
        if product_ids[i] == product_ids[j] and register_ids[i] != register_ids[j]:
            sp_dr_pairs.append((i, j))
        elif product_ids[i] != product_ids[j] and category_ids[i] == category_ids[j]:
            dp_sc_pairs.append((i, j))
        elif category_ids[i] != category_ids[j]:
            dc_pairs.append((i, j))

    results = {}
    for name, pairs in [("SP_DR", sp_dr_pairs), ("DP_SC", dp_sc_pairs), ("DC", dc_pairs)]:
        if not pairs:
            results[name] = np.full(n_layers, np.nan)
            continue

        # Subsample if too many pairs (for DC which can be huge)
        if len(pairs) > 5000:
            rng = np.random.RandomState(42)
            pairs = [pairs[i] for i in rng.choice(len(pairs), 5000, replace=False)]

        layer_sims = np.zeros(n_layers)
        for layer in range(n_layers):
            sims = []
            for i, j in pairs:
                vi = hidden_states[i, layer]
                vj = hidden_states[j, layer]
                cos_sim = np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-10)
                sims.append(cos_sim)
            layer_sims[layer] = np.mean(sims)

        results[name] = layer_sims

    return results
```

### Step 3.5: Entry Point

**File: `scripts/run_stage3.py`**

```python
"""Stage 3: Similarity Analysis.

Usage:
    python scripts/run_stage3.py --config config/default.yaml --model primary
"""

import argparse
import json
import h5py
import numpy as np
from pathlib import Path

from plh.config import load_config
from plh.stage1_stimuli.schema import StimulusDataset
from plh.stage3_similarity.rdm import compute_rdms_all_layers, build_model_rdm
from plh.stage3_similarity.rsa import rsa_all_layers, rsa_permutation_test
from plh.stage3_similarity.anisotropy import correct_anisotropy_all_layers
from plh.stage3_similarity.cosine import compute_condition_similarities
from plh.utils.seeds import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Run similarity analysis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["primary", "fp16_subset", "validation"])
    parser.add_argument("--stimuli", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config.experiment.seed)

    # Load stimuli metadata
    stimuli_path = args.stimuli or Path(config.experiment.output_dir) / "stimuli" / "stimuli.json"
    with open(stimuli_path) as f:
        stimuli = StimulusDataset(**json.load(f))

    # Load hidden states
    hs_path = Path(config.experiment.output_dir) / "hidden_states" / f"{args.model}_states.h5"
    with h5py.File(hs_path, "r") as f:
        stimulus_ids = list(f["stimulus_ids"].asstr()[:])
        hidden_states = f["hidden_states_mean_no_special"][:]  # (N, L, D)

    # Build index mapping stimulus_id -> metadata
    stim_map = {s.stimulus_id: s for s in stimuli.stimuli}
    product_ids = [stim_map[sid].product_id for sid in stimulus_ids]
    category_ids = [stim_map[sid].category for sid in stimulus_ids]
    register_ids = [stim_map[sid].register for sid in stimulus_ids]

    output_dir = Path(config.experiment.output_dir) / "similarity" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run for each anisotropy correction method
    for correction in config.similarity.anisotropy_correction:
        print(f"\n=== Anisotropy correction: {correction} ===")

        corrected = correct_anisotropy_all_layers(hidden_states, method=correction)

        # 1. RDMs
        print("Computing RDMs...")
        observed_rdms = compute_rdms_all_layers(corrected)

        # 2. Model RDM
        model_rdm = build_model_rdm(
            stimulus_ids, product_ids, category_ids,
            **config.similarity.model_rdm,
        )

        # 3. RSA
        print("Running RSA...")
        rsa_corrs, rsa_pvals = rsa_all_layers(observed_rdms, model_rdm)

        # 4. Permutation test at peak layer
        peak_layer = np.argmax(rsa_corrs)
        print(f"Peak RSA at layer {peak_layer}: r={rsa_corrs[peak_layer]:.4f}")
        obs_corr, perm_pval, null_dist = rsa_permutation_test(
            observed_rdms[peak_layer], model_rdm
        )
        print(f"Permutation test p-value: {perm_pval:.6f}")

        # 5. Condition similarities
        print("Computing condition similarities...")
        cond_sims = compute_condition_similarities(
            corrected, product_ids, category_ids, register_ids
        )

        # Save results
        prefix = f"{correction}"
        np.save(output_dir / f"{prefix}_rsa_correlations.npy", rsa_corrs)
        np.save(output_dir / f"{prefix}_rsa_pvalues.npy", rsa_pvals)
        np.save(output_dir / f"{prefix}_permutation_null.npy", null_dist)
        for cond_name, sims in cond_sims.items():
            np.save(output_dir / f"{prefix}_{cond_name}_similarities.npy", sims)

        # Save summary
        summary = {
            "correction": correction,
            "peak_rsa_layer": int(peak_layer),
            "peak_rsa_correlation": float(rsa_corrs[peak_layer]),
            "peak_rsa_pvalue": float(perm_pval),
            "n_stimuli": len(stimulus_ids),
            "n_layers": len(rsa_corrs),
        }
        with open(output_dir / f"{prefix}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print("\nStage 3 complete.")


if __name__ == "__main__":
    main()
```

### Stage 3 Acceptance Criteria

1. RSA correlations computed for all layers, all three anisotropy correction methods
2. Permutation test p-value computed at peak layer
3. Condition similarities (SP-DR, DP-SC, DC) computed per layer
4. All outputs saved as .npy files with JSON summaries
5. No NaN values in RSA correlations (handle edge cases)
6. Condition similarity curves show expected ordering at *some* layers (SP-DR > DP-SC > DC) -- not required to hold everywhere, but if reversed everywhere, that is a flag

---

## 7. Stage 4: Linear Probes

### Step 4.1: Probe Training

**File: `src/plh/stage4_probes/train.py`**

```python
"""Linear probe training pipeline.

Trains L2-regularized logistic regression probes at each layer for:
- 40-class product identification (primary)
- 8-class category classification
- 5-class register classification

Uses stratified K-fold cross-validation.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass


@dataclass
class ProbeResult:
    """Result of training a probe at one layer."""
    layer: int
    task: str
    macro_f1_mean: float
    macro_f1_std: float
    macro_f1_ci_low: float
    macro_f1_ci_high: float
    per_fold_f1: list[float]
    n_classes: int
    n_samples: int


def train_probe_at_layer(
    representations: NDArray,  # (N, D)
    labels: NDArray,           # (N,) integer labels
    layer: int,
    task_name: str,
    n_folds: int = 5,
    regularization: float = 1.0,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> ProbeResult:
    """Train and evaluate a linear probe at one layer.

    Args:
        representations: (N, D) stimulus representations at this layer
        labels: (N,) integer class labels
        layer: layer index (for bookkeeping)
        task_name: probe task name
        n_folds: number of CV folds
        regularization: L2 regularization C parameter
        bootstrap_n: number of bootstrap samples for CI
        seed: random seed

    Returns:
        ProbeResult with macro F1 stats
    """
    from sklearn.metrics import f1_score

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_f1s = []

    for train_idx, test_idx in skf.split(representations, labels):
        X_train, X_test = representations[train_idx], representations[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(
            C=regularization,
            max_iter=2000,
            solver="lbfgs",
            multi_class="multinomial",
            random_state=seed,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        fold_f1s.append(f1)

    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)

    # Bootstrap CI
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(bootstrap_n):
        boot_sample = rng.choice(fold_f1s, size=len(fold_f1s), replace=True)
        boot_means.append(np.mean(boot_sample))
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)

    return ProbeResult(
        layer=layer,
        task=task_name,
        macro_f1_mean=mean_f1,
        macro_f1_std=std_f1,
        macro_f1_ci_low=ci_low,
        macro_f1_ci_high=ci_high,
        per_fold_f1=fold_f1s,
        n_classes=len(np.unique(labels)),
        n_samples=len(labels),
    )


def train_probes_all_layers(
    hidden_states: NDArray,  # (N, L, D)
    product_labels: NDArray,
    category_labels: NDArray,
    register_labels: NDArray,
    config_probes,  # ProbesConfig
) -> list[ProbeResult]:
    """Train all probe tasks at all layers.

    Returns list of ProbeResult, one per (layer, task) combination.
    """
    from tqdm import tqdm

    n_stimuli, n_layers, hidden_dim = hidden_states.shape
    all_results = []

    tasks = {
        "product_40class": product_labels,
        "category_8class": category_labels,
        "register_5class": register_labels,
    }

    for task_name, labels in tasks.items():
        print(f"\nTraining probe: {task_name}")
        for layer in tqdm(range(n_layers), desc=task_name):
            result = train_probe_at_layer(
                hidden_states[:, layer, :],
                labels,
                layer=layer,
                task_name=task_name,
                n_folds=config_probes.cv_folds,
                regularization=config_probes.regularization,
                bootstrap_n=config_probes.bootstrap_n,
            )
            all_results.append(result)

    return all_results
```

### Step 4.2: Zone Classification

**File: `src/plh/stage4_probes/zone_classifier.py`**

```python
"""Zone-based classification comparison.

Trains probes on representations averaged across zone layers:
- Early: layers 0-5
- Protocol: middle 60% of layer stack
- Late: last 10% of layer stack
- Output: final layer only
"""

import numpy as np
from numpy.typing import NDArray
from plh.stage4_probes.train import train_probe_at_layer, ProbeResult
from plh.config import ZonesConfig


def compute_zone_boundaries(n_layers: int, zones_config: ZonesConfig) -> dict[str, tuple[int, int]]:
    """Compute layer index boundaries for each zone.

    Returns dict mapping zone name to (start, end) inclusive layer indices.
    """
    return {
        "early": (zones_config.early[0], zones_config.early[1]),
        "protocol": (
            int(n_layers * zones_config.protocol_pct[0]),
            int(n_layers * zones_config.protocol_pct[1]),
        ),
        "late": (
            int(n_layers * zones_config.late_pct[0]),
            int(n_layers * zones_config.late_pct[1]),
        ),
        "output": (n_layers - 1, n_layers - 1),  # Single layer
    }


def train_zone_probes(
    hidden_states: NDArray,  # (N, L, D)
    labels: NDArray,
    task_name: str,
    zones_config: ZonesConfig,
    n_folds: int = 5,
    regularization: float = 1.0,
) -> dict[str, ProbeResult]:
    """Train a probe for each zone using mean-pooled zone representations.

    Returns dict mapping zone name to ProbeResult.
    """
    n_stimuli, n_layers, hidden_dim = hidden_states.shape
    zones = compute_zone_boundaries(n_layers, zones_config)
    results = {}

    for zone_name, (start, end) in zones.items():
        # Mean pool across layers in this zone
        zone_repr = hidden_states[:, start:end+1, :].mean(axis=1)  # (N, D)

        result = train_probe_at_layer(
            zone_repr,
            labels,
            layer=-1,  # Indicates zone, not single layer
            task_name=f"{task_name}_zone_{zone_name}",
            n_folds=n_folds,
            regularization=regularization,
        )
        results[zone_name] = result

    return results
```

### Step 4.3: Entry Point

**File: `scripts/run_stage4.py`**

```python
"""Stage 4: Linear Probes.

Usage:
    python scripts/run_stage4.py --config config/default.yaml --model primary
"""

import argparse
import json
import h5py
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

from plh.config import load_config
from plh.stage1_stimuli.schema import StimulusDataset
from plh.stage3_similarity.anisotropy import correct_anisotropy_all_layers
from plh.stage4_probes.train import train_probes_all_layers
from plh.stage4_probes.zone_classifier import train_zone_probes
from plh.utils.seeds import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Train linear probes")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=["primary", "fp16_subset", "validation"])
    parser.add_argument("--correction", default="none",
                        choices=["none", "mean_centering", "whitening"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config.experiment.seed)

    # Load stimuli and hidden states
    stimuli_path = Path(config.experiment.output_dir) / "stimuli" / "stimuli.json"
    with open(stimuli_path) as f:
        stimuli = StimulusDataset(**json.load(f))

    hs_path = Path(config.experiment.output_dir) / "hidden_states" / f"{args.model}_states.h5"
    with h5py.File(hs_path, "r") as f:
        stimulus_ids = list(f["stimulus_ids"].asstr()[:])
        hidden_states = f["hidden_states_mean_no_special"][:]

    # Apply anisotropy correction
    hidden_states = correct_anisotropy_all_layers(hidden_states, method=args.correction)

    # Build label arrays aligned with stimulus_ids in HDF5
    stim_map = {s.stimulus_id: s for s in stimuli.stimuli}

    product_enc = LabelEncoder()
    category_enc = LabelEncoder()
    register_enc = LabelEncoder()

    products = [stim_map[sid].product_id for sid in stimulus_ids]
    categories = [stim_map[sid].category for sid in stimulus_ids]
    registers = [stim_map[sid].register for sid in stimulus_ids]

    product_labels = product_enc.fit_transform(products)
    category_labels = category_enc.fit_transform(categories)
    register_labels = register_enc.fit_transform(registers)

    # Per-layer probes
    print("Training per-layer probes...")
    results = train_probes_all_layers(
        hidden_states, product_labels, category_labels, register_labels, config.probes
    )

    # Zone probes
    print("Training zone probes...")
    zone_results = {}
    for task_name, labels in [
        ("product", product_labels),
        ("category", category_labels),
        ("register", register_labels),
    ]:
        zone_results[task_name] = train_zone_probes(
            hidden_states, labels, task_name, config.zones,
            config.probes.cv_folds, config.probes.regularization,
        )

    # Save
    output_dir = Path(config.experiment.output_dir) / "probes" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-layer results as JSON
    layer_results = [
        {
            "layer": r.layer,
            "task": r.task,
            "macro_f1_mean": r.macro_f1_mean,
            "macro_f1_std": r.macro_f1_std,
            "macro_f1_ci_low": r.macro_f1_ci_low,
            "macro_f1_ci_high": r.macro_f1_ci_high,
            "per_fold_f1": r.per_fold_f1,
            "n_classes": r.n_classes,
            "n_samples": r.n_samples,
        }
        for r in results
    ]
    with open(output_dir / f"{args.correction}_layer_probes.json", "w") as f:
        json.dump(layer_results, f, indent=2)

    # Zone results
    zone_output = {}
    for task, zones in zone_results.items():
        zone_output[task] = {}
        for zone_name, r in zones.items():
            zone_output[task][zone_name] = {
                "macro_f1_mean": r.macro_f1_mean,
                "macro_f1_ci_low": r.macro_f1_ci_low,
                "macro_f1_ci_high": r.macro_f1_ci_high,
            }
    with open(output_dir / f"{args.correction}_zone_probes.json", "w") as f:
        json.dump(zone_output, f, indent=2)

    print(f"Stage 4 complete. Results in {output_dir}")


if __name__ == "__main__":
    main()
```

### Stage 4 Acceptance Criteria

1. Per-layer probe results for all three tasks at all layers
2. Zone probe results for all four zones x three tasks
3. Bootstrap CIs computed and non-degenerate
4. 40-class product probe achieves above chance (>2.5%) at peak layer
5. Results saved as JSON with all metrics
6. Category probe accuracy > register probe accuracy at some middle layers (directional check, not pass/fail)

---

## 8. Stage 5: Analysis & Reporting

### Step 5.1: Hypothesis Tests

**File: `src/plh/stage5_analysis/hypothesis_tests.py`**

```python
"""Pre-registered hypothesis tests against falsification criteria.

H1: Phase structure -- SP-DR similarity increases from early to middle layers
H2: Content dominance -- category probe > register probe in protocol zone
H3: Protocol advantage -- protocol zone F1 > output layer F1
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class HypothesisResult:
    hypothesis: str
    supported: bool
    effect_size: float
    p_value: float
    criterion: str
    observed_value: str
    detail: str


def test_h1_phase_structure(
    sp_dr_similarities: np.ndarray,  # (L,) per-layer mean SP-DR similarity
    early_layers: tuple[int, int] = (0, 5),
    mid_start_pct: float = 0.3,
    mid_end_pct: float = 0.7,
    alpha: float = 0.05,
) -> HypothesisResult:
    """H1: SP-DR similarity significantly increases from early to middle layers.

    Falsified if p > 0.05 (paired t-test on early vs. mid layer similarities).
    """
    n_layers = len(sp_dr_similarities)
    mid_start = int(n_layers * mid_start_pct)
    mid_end = int(n_layers * mid_end_pct)

    early_sims = sp_dr_similarities[early_layers[0]:early_layers[1]+1]
    mid_sims = sp_dr_similarities[mid_start:mid_end+1]

    # Use means for the test since arrays may differ in length
    # Better: compare each early layer to its corresponding mid-layer position
    # Simple approach: two-sample t-test
    t_stat, p_value = stats.ttest_ind(mid_sims, early_sims, alternative="greater")
    effect = np.mean(mid_sims) - np.mean(early_sims)

    return HypothesisResult(
        hypothesis="H1: Phase Structure",
        supported=p_value < alpha,
        effect_size=effect,
        p_value=p_value,
        criterion=f"p < {alpha} (one-tailed t-test, mid > early SP-DR similarity)",
        observed_value=f"early mean={np.mean(early_sims):.4f}, mid mean={np.mean(mid_sims):.4f}",
        detail=f"t={t_stat:.3f}, p={p_value:.6f}, effect={effect:.4f}",
    )


def test_h2_content_dominance(
    category_f1s: np.ndarray,   # (L,) per-layer category probe F1
    register_f1s: np.ndarray,   # (L,) per-layer register probe F1
    protocol_zone: tuple[int, int],
    margin: float = 5.0,        # percentage points
) -> HypothesisResult:
    """H2: In the protocol zone, category probe > register probe.

    Falsified if register > category by more than 'margin' pp at any protocol layer.
    """
    pz_cat = category_f1s[protocol_zone[0]:protocol_zone[1]+1]
    pz_reg = register_f1s[protocol_zone[0]:protocol_zone[1]+1]

    # Check: does register exceed category by > margin at ANY protocol layer?
    diffs = (pz_reg - pz_cat) * 100  # Convert to percentage points
    max_register_advantage = np.max(diffs)

    supported = max_register_advantage <= margin
    mean_cat = np.mean(pz_cat) * 100
    mean_reg = np.mean(pz_reg) * 100

    return HypothesisResult(
        hypothesis="H2: Content Dominance",
        supported=supported,
        effect_size=mean_cat - mean_reg,
        p_value=float("nan"),  # Not applicable (criterion-based)
        criterion=f"Register probe does NOT exceed category probe by >{margin}pp in protocol zone",
        observed_value=f"protocol zone: cat={mean_cat:.1f}%, reg={mean_reg:.1f}%, max reg advantage={max_register_advantage:.1f}pp",
        detail=f"Category dominates in {np.sum(diffs < 0)}/{len(diffs)} protocol layers",
    )


def test_h3_protocol_advantage(
    zone_f1s: dict[str, float],  # {"early": f1, "protocol": f1, "late": f1, "output": f1}
    margin: float = 2.0,
) -> HypothesisResult:
    """H3: Protocol zone F1 > output layer F1.

    Falsified if best layer is outside middle 60%, or if protocol zone
    does not outperform output by at least 'margin' pp.
    """
    protocol_f1 = zone_f1s["protocol"] * 100
    output_f1 = zone_f1s["output"] * 100
    advantage = protocol_f1 - output_f1

    supported = advantage >= margin

    return HypothesisResult(
        hypothesis="H3: Protocol Layer Advantage",
        supported=supported,
        effect_size=advantage,
        p_value=float("nan"),
        criterion=f"Protocol zone F1 exceeds output F1 by >= {margin}pp",
        observed_value=f"protocol={protocol_f1:.1f}%, output={output_f1:.1f}%, advantage={advantage:.1f}pp",
        detail=f"All zones: {', '.join(f'{k}={v*100:.1f}%' for k,v in zone_f1s.items())}",
    )
```

### Step 5.2: Control Analyses

**File: `src/plh/stage5_analysis/controls.py`**

```python
"""Control analyses: memorization, quantization, generator effects."""

import numpy as np
from scipy.stats import spearmanr
from dataclasses import dataclass


@dataclass
class ControlResult:
    name: str
    passed: bool
    metric: float
    threshold: float
    detail: str


def memorization_control(
    real_rsa_correlations: np.ndarray,     # (L,) RSA for real products
    fictional_rsa_correlations: np.ndarray, # (L,) RSA for fictional products
) -> ControlResult:
    """Compare RSA curves for real vs. fictional products.

    If fictional products show similar phase structure to real products,
    the effect is not explained by memorization of brand-specific information.
    """
    corr, pval = spearmanr(real_rsa_correlations, fictional_rsa_correlations)
    return ControlResult(
        name="Memorization Control (Real vs. Fictional)",
        passed=corr > 0.7,  # High correlation = similar phase structure
        metric=corr,
        threshold=0.7,
        detail=f"Spearman r={corr:.4f}, p={pval:.6f}. "
               f"Real peak={np.argmax(real_rsa_correlations)}, "
               f"Fictional peak={np.argmax(fictional_rsa_correlations)}",
    )


def quantization_control(
    fp16_rsa_correlations: np.ndarray,    # (L,) RSA at FP16
    quant_rsa_correlations: np.ndarray,   # (L,) RSA at 4-bit
    threshold: float = 0.9,
) -> ControlResult:
    """Compare RSA curves between FP16 and quantized model.

    Pre-registered criterion: Spearman > 0.9 between per-layer RSA curves.
    """
    # May need to align layer indices if FP16 subset has different coverage
    min_len = min(len(fp16_rsa_correlations), len(quant_rsa_correlations))
    corr, pval = spearmanr(
        fp16_rsa_correlations[:min_len],
        quant_rsa_correlations[:min_len]
    )
    return ControlResult(
        name="Quantization Control (FP16 vs. 4-bit)",
        passed=corr > threshold,
        metric=corr,
        threshold=threshold,
        detail=f"Spearman r={corr:.4f}, p={pval:.6f}",
    )


def generator_control(
    claude_rsa_at_peak: float,
    gpt4_rsa_at_peak: float,
    tolerance: float = 0.15,  # Max acceptable difference
) -> ControlResult:
    """Compare RSA at peak layer across generators.

    If Claude and GPT-4 generated stimuli produce similar RSA at peak,
    the effect is not generator-specific.
    """
    diff = abs(claude_rsa_at_peak - gpt4_rsa_at_peak)
    return ControlResult(
        name="Generator Control (Claude vs. GPT-4)",
        passed=diff < tolerance,
        metric=diff,
        threshold=tolerance,
        detail=f"Claude RSA={claude_rsa_at_peak:.4f}, GPT-4 RSA={gpt4_rsa_at_peak:.4f}, diff={diff:.4f}",
    )
```

### Step 5.3: Go/No-Go Decision

**File: `src/plh/stage5_analysis/go_no_go.py`**

```python
"""Go/no-go assessment for the protocol layer hypothesis."""

from dataclasses import dataclass
from plh.stage5_analysis.hypothesis_tests import HypothesisResult
from plh.stage5_analysis.controls import ControlResult


@dataclass
class GoNoGoDecision:
    verdict: str  # "GO", "QUALIFIED_GO", "NO_GO"
    hypotheses: list[HypothesisResult]
    controls: list[ControlResult]
    summary: str


def assess(
    h1: HypothesisResult,
    h2: HypothesisResult,
    h3: HypothesisResult,
    controls: list[ControlResult],
) -> GoNoGoDecision:
    """Evaluate all results and produce a go/no-go decision.

    GO: All three hypotheses supported, all controls passed.
    QUALIFIED_GO: H1+H2 supported with controls, H3 marginal or one control marginal.
    NO_GO: Any hypothesis falsified, or critical control failed.
    """
    h_results = [h1, h2, h3]
    all_h_supported = all(h.supported for h in h_results)
    critical_controls = [c for c in controls if "Quantization" in c.name or "Memorization" in c.name]
    all_critical_passed = all(c.passed for c in critical_controls)

    if all_h_supported and all_critical_passed:
        verdict = "GO"
        summary = (
            "All three hypotheses supported with pre-registered criteria met. "
            "All critical controls passed. The protocol layer hypothesis has "
            "preliminary support for further investigation."
        )
    elif h1.supported and h2.supported and all_critical_passed:
        verdict = "QUALIFIED_GO"
        summary = (
            "H1 (phase structure) and H2 (content dominance) supported. "
            f"H3 (protocol advantage): {'supported' if h3.supported else 'NOT supported'}. "
            "Results suggest partial support; further investigation warranted with caveats."
        )
    else:
        verdict = "NO_GO"
        failed_h = [h.hypothesis for h in h_results if not h.supported]
        failed_c = [c.name for c in critical_controls if not c.passed]
        summary = (
            f"Falsified hypotheses: {failed_h or 'none'}. "
            f"Failed controls: {failed_c or 'none'}. "
            "The protocol layer hypothesis is not supported by this experiment."
        )

    return GoNoGoDecision(
        verdict=verdict,
        hypotheses=h_results,
        controls=controls,
        summary=summary,
    )
```

### Step 5.4: Visualization

**File: `src/plh/visualization/phase_plots.py`** (representative example; other viz modules follow the same pattern)

```python
"""Phase structure visualization.

Generates:
1. Three-condition similarity curves (SP-DR, DP-SC, DC) across layers
2. RSA correlation across layers
3. Combined phase structure + probe accuracy plot
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import numpy as np
from pathlib import Path


def plot_condition_similarities(
    sp_dr: np.ndarray,
    dp_sc: np.ndarray,
    dc: np.ndarray,
    title: str = "Per-Layer Cosine Similarity by Condition",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot the three-condition similarity curves.

    The signature phase structure pattern would show:
    - SP-DR increasing to peak in middle layers, then decreasing
    - DP-SC following similar but lower trajectory
    - DC remaining relatively flat or lower
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = np.arange(len(sp_dr))

    ax.plot(layers, sp_dr, label="Same Product, Diff Register (SP-DR)", color="#2196F3", linewidth=2)
    ax.plot(layers, dp_sc, label="Diff Product, Same Category (DP-SC)", color="#FF9800", linewidth=2)
    ax.plot(layers, dc, label="Different Category (DC)", color="#F44336", linewidth=2)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_rsa_across_layers(
    rsa_corrs: np.ndarray,
    rsa_pvals: np.ndarray,
    title: str = "RSA Correlation with Semantic Model RDM",
    alpha: float = 0.05,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot RSA correlation curve with significance markers."""
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = np.arange(len(rsa_corrs))

    ax.plot(layers, rsa_corrs, color="#4CAF50", linewidth=2, label="RSA (Spearman)")

    # Mark significant layers
    sig_mask = rsa_pvals < alpha
    ax.scatter(layers[sig_mask], rsa_corrs[sig_mask], color="#4CAF50", s=20, zorder=5)
    ax.scatter(layers[~sig_mask], rsa_corrs[~sig_mask], color="gray", s=10, alpha=0.5, zorder=5)

    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Spearman Correlation", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_probe_accuracy_curves(
    product_f1s: np.ndarray,
    category_f1s: np.ndarray,
    register_f1s: np.ndarray,
    title: str = "Linear Probe Accuracy Across Layers",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot probe accuracy curves for all three tasks."""
    fig, ax = plt.subplots(figsize=(12, 6))
    layers = np.arange(len(product_f1s))

    ax.plot(layers, product_f1s * 100, label="Product (40-class)", color="#9C27B0", linewidth=2)
    ax.plot(layers, category_f1s * 100, label="Category (8-class)", color="#2196F3", linewidth=2)
    ax.plot(layers, register_f1s * 100, label="Register (5-class)", color="#FF5722", linewidth=2)

    # Chance levels
    ax.axhline(y=2.5, color="#9C27B0", linestyle=":", alpha=0.3, label="Chance (product)")
    ax.axhline(y=12.5, color="#2196F3", linestyle=":", alpha=0.3, label="Chance (category)")
    ax.axhline(y=20.0, color="#FF5722", linestyle=":", alpha=0.3, label="Chance (register)")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Macro F1 (%)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
```

### Step 5.5: Reporting Entry Point

**File: `scripts/run_stage5.py`**

```python
"""Stage 5: Analysis & Reporting.

Runs all hypothesis tests, controls, generates visualizations,
and produces the final go/no-go report.

Usage:
    python scripts/run_stage5.py --config config/default.yaml
"""

import argparse
import json
import numpy as np
from pathlib import Path

from plh.config import load_config
from plh.stage5_analysis.hypothesis_tests import test_h1_phase_structure, test_h2_content_dominance, test_h3_protocol_advantage
from plh.stage5_analysis.controls import memorization_control, quantization_control, generator_control
from plh.stage5_analysis.go_no_go import assess
from plh.visualization.phase_plots import plot_condition_similarities, plot_rsa_across_layers, plot_probe_accuracy_curves
from plh.stage4_probes.zone_classifier import compute_zone_boundaries
from plh.utils.seeds import set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Run final analysis and reporting")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_global_seed(config.experiment.seed)
    base = Path(config.experiment.output_dir)

    report_dir = base / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Load Stage 3 results (primary model, no correction and corrected)
    sim_dir = base / "similarity" / "primary"
    for correction in config.similarity.anisotropy_correction:
        sp_dr = np.load(sim_dir / f"{correction}_SP_DR_similarities.npy")
        dp_sc = np.load(sim_dir / f"{correction}_DP_SC_similarities.npy")
        dc = np.load(sim_dir / f"{correction}_DC_similarities.npy")
        rsa_corrs = np.load(sim_dir / f"{correction}_rsa_correlations.npy")
        rsa_pvals = np.load(sim_dir / f"{correction}_rsa_pvalues.npy")

        # Visualizations
        plot_condition_similarities(sp_dr, dp_sc, dc,
            title=f"Condition Similarities ({correction})",
            output_path=fig_dir / f"conditions_{correction}.png")
        plot_rsa_across_layers(rsa_corrs, rsa_pvals,
            title=f"RSA Correlation ({correction})",
            output_path=fig_dir / f"rsa_{correction}.png")

    # Load Stage 4 results
    probe_dir = base / "probes" / "primary"
    # Use the "none" (uncorrected) as primary for hypothesis tests,
    # report both as sensitivity analysis
    with open(probe_dir / "none_layer_probes.json") as f:
        layer_probes = json.load(f)
    with open(probe_dir / "none_zone_probes.json") as f:
        zone_probes = json.load(f)

    # Extract per-layer arrays
    n_layers = max(r["layer"] for r in layer_probes) + 1
    product_f1s = np.zeros(n_layers)
    category_f1s = np.zeros(n_layers)
    register_f1s = np.zeros(n_layers)
    for r in layer_probes:
        if "product" in r["task"]:
            product_f1s[r["layer"]] = r["macro_f1_mean"]
        elif "category" in r["task"]:
            category_f1s[r["layer"]] = r["macro_f1_mean"]
        elif "register" in r["task"]:
            register_f1s[r["layer"]] = r["macro_f1_mean"]

    plot_probe_accuracy_curves(product_f1s, category_f1s, register_f1s,
        output_path=fig_dir / "probe_curves.png")

    # Hypothesis tests
    sp_dr_none = np.load(sim_dir / "none_SP_DR_similarities.npy")
    zones = compute_zone_boundaries(n_layers, config.zones)

    h1 = test_h1_phase_structure(sp_dr_none,
        early_layers=tuple(config.zones.early),
        mid_start_pct=config.zones.protocol_pct[0],
        mid_end_pct=config.zones.protocol_pct[1])

    h2 = test_h2_content_dominance(category_f1s, register_f1s,
        protocol_zone=(zones["protocol"][0], zones["protocol"][1]),
        margin=config.analysis.h2_register_dominance_margin)

    product_zone_f1s = {z: zone_probes["product"][z]["macro_f1_mean"] for z in zone_probes["product"]}
    h3 = test_h3_protocol_advantage(product_zone_f1s,
        margin=config.analysis.h3_protocol_advantage_margin)

    # Controls (load comparison data)
    controls_list = []
    # ... (load FP16 subset RSA, fictional-only RSA, generator-specific RSA)
    # ... (run memorization_control, quantization_control, generator_control)

    # Go/No-Go
    decision = assess(h1, h2, h3, controls_list)

    # Save final report
    report = {
        "verdict": decision.verdict,
        "summary": decision.summary,
        "hypotheses": [
            {"name": h.hypothesis, "supported": h.supported, "detail": h.detail}
            for h in decision.hypotheses
        ],
        "controls": [
            {"name": c.name, "passed": c.passed, "metric": c.metric, "detail": c.detail}
            for c in decision.controls
        ],
    }
    with open(report_dir / "final_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"VERDICT: {decision.verdict}")
    print(f"{'='*60}")
    print(decision.summary)
    print(f"\nFull report: {report_dir / 'final_report.json'}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()
```

---

## 9. Testing Strategy

All tests run without GPU, API keys, or network. Mock all external dependencies.

### Fixtures (`tests/conftest.py`)

```python
"""Shared test fixtures."""

import pytest
import numpy as np
from plh.stage1_stimuli.schema import Stimulus, StimulusDataset
from plh.constants import Category, Register


@pytest.fixture
def tiny_hidden_states():
    """(10 stimuli, 4 layers, 8 dimensions) -- tiny for fast tests."""
    rng = np.random.RandomState(42)
    return rng.randn(10, 4, 8).astype(np.float32)


@pytest.fixture
def mock_stimuli():
    """10 mock stimuli: 2 categories x 2 products x 2 registers + extras."""
    stimuli = []
    for cat_idx, cat in enumerate(["oral_care", "pet_food"]):
        for prod_idx in range(2):
            for reg in ["marketing", "social"]:
                for variant in range(2):
                    stimuli.append(Stimulus(
                        stimulus_id=f"{cat}_prod{prod_idx}_{reg}_{variant}_anthropic",
                        product_id=f"{cat}_prod{prod_idx}",
                        product_name=f"Test Product {cat_idx}_{prod_idx}",
                        category=cat,
                        register=reg,
                        variant=variant,
                        text="This is a test product description with sufficient length for testing.",
                        token_count=85,
                        core_attributes_present=["attr1", "attr2"],
                        is_fictional=False,
                        generator="anthropic",
                    ))
    # Trim to 10
    return stimuli[:10]


@pytest.fixture
def mock_dataset(mock_stimuli):
    return StimulusDataset(
        version="test",
        generation_date="2026-01-01",
        config_hash="test",
        stimuli=mock_stimuli,
        metadata={"n_stimuli": len(mock_stimuli)},
    )
```

### Key Tests

| Test File | What It Tests | Mock Strategy |
|-----------|--------------|---------------|
| `test_config.py` | Config loading, validation, merging | No mocks needed |
| `test_stage1/test_schema.py` | Stimulus Pydantic model validation | No mocks |
| `test_stage1/test_prompts.py` | Prompt generation produces valid strings | No mocks |
| `test_stage1/test_validate.py` | Token counting, attribute extraction | Mock tokenizer |
| `test_stage2/test_hooks.py` | Hook registration/capture on tiny model | Use `nn.Linear` mock |
| `test_stage2/test_pooling.py` | Mean pooling, last-token pooling | Synthetic tensors |
| `test_stage2/test_extract.py` | HDF5 save/load, checkpoint resume | Mock model, synthetic data |
| `test_stage3/test_rdm.py` | RDM computation, model RDM construction | Synthetic vectors |
| `test_stage3/test_rsa.py` | RSA correlation, permutation test | Synthetic RDMs |
| `test_stage3/test_anisotropy.py` | Centering, whitening, dimension checks | Synthetic data |
| `test_stage4/test_train.py` | Probe training, CV, F1 computation | Synthetic embeddings |
| `test_stage4/test_evaluate.py` | Bootstrap CI computation | Synthetic F1 arrays |
| `test_stage5/test_hypothesis_tests.py` | H1/H2/H3 test logic | Synthetic curves |
| `test_stage5/test_controls.py` | Control comparison logic | Synthetic RSA curves |

### Example Test

```python
# tests/test_stage3/test_rdm.py

import numpy as np
import pytest
from plh.stage3_similarity.rdm import compute_rdm, build_model_rdm


def test_rdm_shape():
    reps = np.random.randn(10, 8).astype(np.float32)
    rdm = compute_rdm(reps)
    assert rdm.shape == (10, 10)


def test_rdm_diagonal_zero():
    reps = np.random.randn(10, 8).astype(np.float32)
    rdm = compute_rdm(reps)
    np.testing.assert_allclose(np.diag(rdm), 0.0, atol=1e-6)


def test_rdm_symmetric():
    reps = np.random.randn(10, 8).astype(np.float32)
    rdm = compute_rdm(reps)
    np.testing.assert_allclose(rdm, rdm.T, atol=1e-6)


def test_model_rdm_structure():
    stim_ids = ["a_m", "a_s", "b_m", "c_m"]
    prod_ids = ["a", "a", "b", "c"]
    cat_ids  = ["x", "x", "x", "y"]
    rdm = build_model_rdm(stim_ids, prod_ids, cat_ids)
    assert rdm[0, 1] == 0.0   # Same product
    assert rdm[0, 2] == 0.5   # Same category, diff product
    assert rdm[0, 3] == 1.0   # Different category
```

### Running Tests

```bash
# All tests (no GPU, no API keys)
pytest tests/ -v

# With coverage
pytest tests/ --cov=plh --cov-report=term-missing

# Skip integration tests
pytest tests/ -m "not integration"
```

---

## 10. Dependency & Execution Order

### Dependency Graph

```
Stage 0 (Scaffolding)
    |
    v
Stage 1 (Stimulus Generation)  -- requires API keys
    |
    v
Stage 2 (Hidden State Extraction)  -- requires GPU, model downloads
    |                                   Run for each model: primary, fp16_subset, validation
    |                                   Models can run in parallel if VRAM allows (they don't)
    v
Stage 3 (Similarity Analysis)  -- CPU only, depends on Stage 2 output
    |                              Can start as soon as one model's extraction is done
    v
Stage 4 (Linear Probes)  -- CPU only, depends on Stage 2 output
    |                        Can run in parallel with Stage 3
    v
Stage 5 (Analysis & Reporting)  -- depends on Stages 3 and 4
```

### Implementation Order for the Coding Agent

**Group 1: Foundation (commit after completing)**
1. Create `pyproject.toml`, `.gitignore`, `.env.example`
2. Create `src/plh/__init__.py`, `src/plh/config.py`, `config/default.yaml`, `config/debug.yaml`
3. Create `src/plh/constants.py` (full product catalog -- 80 products)
4. Create `src/plh/utils/seeds.py`, `src/plh/utils/io.py`, `src/plh/utils/checkpoint.py`
5. Create `tests/conftest.py`, `tests/test_config.py`
6. Run: `pytest tests/test_config.py`

**Group 2: Stage 1 (commit after completing)**
7. Create `src/plh/stage1_stimuli/schema.py`
8. Create `src/plh/stage1_stimuli/prompts.py`
9. Create `src/plh/stage1_stimuli/validate.py`
10. Create `src/plh/stage1_stimuli/generate.py`
11. Create `scripts/run_stage1.py`
12. Create `tests/test_stage1/` (all test files)
13. Run: `pytest tests/test_stage1/`

**Group 3: Stage 2 (commit after completing)**
14. Create `src/plh/stage2_extraction/models.py`
15. Create `src/plh/stage2_extraction/hooks.py`
16. Create `src/plh/stage2_extraction/pooling.py`
17. Create `src/plh/stage2_extraction/extract.py`
18. Create `scripts/run_stage2.py`
19. Create `tests/test_stage2/` (all test files)
20. Run: `pytest tests/test_stage2/`

**Group 4: Stage 3 (commit after completing)**
21. Create `src/plh/stage3_similarity/rdm.py`
22. Create `src/plh/stage3_similarity/anisotropy.py`
23. Create `src/plh/stage3_similarity/rsa.py`
24. Create `src/plh/stage3_similarity/cosine.py`
25. Create `scripts/run_stage3.py`
26. Create `tests/test_stage3/` (all test files)
27. Run: `pytest tests/test_stage3/`

**Group 5: Stage 4 (commit after completing)**
28. Create `src/plh/stage4_probes/train.py`
29. Create `src/plh/stage4_probes/evaluate.py`
30. Create `src/plh/stage4_probes/zone_classifier.py`
31. Create `scripts/run_stage4.py`
32. Create `tests/test_stage4/` (all test files)
33. Run: `pytest tests/test_stage4/`

**Group 6: Stage 5 + Visualization (commit after completing)**
34. Create `src/plh/stage5_analysis/hypothesis_tests.py`
35. Create `src/plh/stage5_analysis/controls.py`
36. Create `src/plh/stage5_analysis/go_no_go.py`
37. Create `src/plh/visualization/phase_plots.py`
38. Create `src/plh/visualization/probe_curves.py`
39. Create `src/plh/visualization/rsa_heatmaps.py`
40. Create `src/plh/visualization/style.py`
41. Create `scripts/run_stage5.py`
42. Create `scripts/run_all.py`
43. Create `scripts/validate_data.py`
44. Create `tests/test_stage5/` (all test files)
45. Run: `pytest tests/`

### Full Pipeline Execution (after all code is written)

```bash
# 1. Install
cd protocol-layer-hypothesis
pip install -e ".[dev]"

# 2. Debug run (no GPU needed for stage 1, small model for stage 2)
python scripts/run_stage1.py --config config/debug.yaml
python scripts/run_stage2.py --config config/debug.yaml --model primary
python scripts/run_stage3.py --config config/debug.yaml --model primary
python scripts/run_stage4.py --config config/debug.yaml --model primary
python scripts/run_stage5.py --config config/debug.yaml

# 3. Full run
python scripts/run_stage1.py --config config/default.yaml
python scripts/run_stage2.py --config config/default.yaml --model primary
python scripts/run_stage2.py --config config/default.yaml --model fp16_subset
python scripts/run_stage2.py --config config/default.yaml --model validation
python scripts/run_stage3.py --config config/default.yaml --model primary
python scripts/run_stage3.py --config config/default.yaml --model fp16_subset
python scripts/run_stage3.py --config config/default.yaml --model validation
python scripts/run_stage4.py --config config/default.yaml --model primary
python scripts/run_stage4.py --config config/default.yaml --model primary --correction whitening
python scripts/run_stage5.py --config config/default.yaml
```

### Estimated Wall-Clock Times

| Stage | Debug Config | Full Config |
|-------|-------------|-------------|
| Stage 1 (stimuli) | 2 min | 30-60 min (API rate limits) |
| Stage 2 (extraction, primary) | 5 min | 3-4 hours |
| Stage 2 (extraction, fp16) | 10 min | 2-3 hours |
| Stage 2 (extraction, validation) | 3 min | 1-2 hours |
| Stage 3 (similarity) | 1 min | 10-20 min |
| Stage 4 (probes) | 1 min | 15-30 min |
| Stage 5 (analysis) | <1 min | 5 min |
| **Total** | **~20 min** | **~8-12 hours** |

---

## Utility Modules

### `src/plh/utils/seeds.py`

```python
"""Reproducibility: global seed management."""

import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### `src/plh/utils/checkpoint.py`

```python
"""Stage-level checkpointing for crash recovery."""

import json
import pickle
from pathlib import Path
from typing import Any


class StageCheckpoint:
    """Simple file-based checkpoint for resumable stages.

    Saves partial results to a pickle file. On restart, loads
    the checkpoint and continues from where it left off.
    """

    def __init__(self, stage_name: str, output_dir: str):
        self.path = Path(output_dir) / ".checkpoints" / f"{stage_name}.pkl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save_partial(self, data: Any) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(data, f)

    def load_partial(self) -> Any | None:
        if self.path.exists():
            with open(self.path, "rb") as f:
                return pickle.load(f)
        return None

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()
```

### `src/plh/utils/io.py`

```python
"""I/O helpers for HDF5 and data validation."""

import h5py
import numpy as np
from pathlib import Path


def validate_hdf5(path: Path, expected_keys: list[str]) -> list[str]:
    """Validate an HDF5 file has expected datasets and no NaN/Inf values.

    Returns list of issues (empty = valid).
    """
    issues = []
    if not path.exists():
        return [f"File not found: {path}"]

    with h5py.File(path, "r") as f:
        for key in expected_keys:
            if key not in f:
                issues.append(f"Missing dataset: {key}")
                continue
            data = f[key][:]
            if np.isnan(data).any():
                issues.append(f"NaN values in {key}: {np.isnan(data).sum()} total")
            if np.isinf(data).any():
                issues.append(f"Inf values in {key}: {np.isinf(data).sum()} total")

    return issues
```

---

## Critical Implementation Notes for the Coding Agent

1. **Model names**: The actual HuggingFace model IDs for GPTQ quantized Qwen models may differ from what is in the config. Before downloading, verify the exact repo name on HuggingFace (e.g., search for "Qwen2.5-27B-GPTQ-Int4" or "Qwen2.5-27B-AWQ"). The config should be updated with the correct repo name once identified.

2. **RTX 5090 vs RTX 3080**: The user's CLAUDE.md says RTX 3080 (10 GB VRAM), but the experiment design says RTX 5090 (32 GB VRAM). **The batch sizes and memory budgets in this plan assume 32 GB VRAM (RTX 5090)**. If the actual GPU is an RTX 3080, the 4-bit Qwen 27B will NOT fit (~14 GB model + activation memory). In that case, switch to `Qwen2.5-7B` as the primary model or use Llama-3.1-8B for everything. **The coding agent should check available VRAM with `nvidia-smi` before choosing model sizes.**

3. **Product catalog**: `constants.py` is shown with 2 example products. The coding agent must expand this to the full 40 real + 40 fictional products. Use the Claude API (or hardcode) to create them. Each product needs 3-5 specific, quantitative core attributes that distinguish it.

4. **FP16 CPU offloading verification**: Before running the full FP16 extraction, the coding agent should do a single-stimulus pilot test to verify that `output_hidden_states=True` returns all layers correctly when the model is split across GPU and CPU. If it fails, fall back to 8-bit quantization instead.

5. **Anisotropy correction in `correct_anisotropy_all_layers`**: The current implementation has a comment noting a potential issue with `np.stack` when whitening changes dimensionality. The coding agent should verify this works correctly or adjust to handle variable output dimensions.

6. **Cross-generator stimuli**: The "human" generator is a stretch goal. The coding agent should implement the infrastructure for it (the schema supports it) but the pipeline should work end-to-end with only "anthropic" and "openai" generators.

7. **`__init__.py` files**: Every `__init__.py` should re-export the module's public API for clean imports. E.g., `from plh.stage3_similarity import compute_rdms_all_layers, rsa_all_layers`.

8. **Error handling**: Every script should catch `KeyboardInterrupt` and save checkpoints before exiting. The extraction pipeline is the most critical -- a crash at stimulus 700/800 should not lose all work.

9. **The `from plh.constants import Category` import in `generate.py`**: Note the constants module also needs to import Category in its own code. Make sure there are no circular imports.

10. **Qwen3.5-27B**: As of the design date, the actual HuggingFace model may be `Qwen/Qwen2.5-27B-Instruct` or similar. The coding agent should search HuggingFace for the correct model name and update configs accordingly. If "Qwen3.5" does not exist as a public model, use `Qwen2.5-27B` as the closest available.
