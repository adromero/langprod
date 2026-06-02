#!/usr/bin/env python3
"""Generate all stimuli using the claude CLI.

Batches by product: each call generates all 5 registers × 2 variants = 10
descriptions per product, for 80 products = 80 CLI calls total.

Saves incrementally to data/stimuli.json after each product.
Supports resume: skips products whose stimuli already exist.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path so we can import stimuli
sys.path.insert(0, str(Path(__file__).parent))

from stimuli import (
    REAL_PRODUCTS,
    FICTIONAL_PRODUCTS,
    REGISTER_SPECS,
    CROSS_GENERATOR_SUBSET_IDS,
    build_generation_prompt,
)

DATA_DIR = Path("data")
STIMULI_FILE = DATA_DIR / "stimuli.json"
REGISTERS = ["marketing", "regulatory", "casual_social", "patent", "journalistic"]


def load_existing() -> list[dict]:
    """Load existing stimuli from disk, or return empty list."""
    if STIMULI_FILE.exists():
        with open(STIMULI_FILE) as f:
            return json.load(f)
    return []


def save_stimuli(stimuli: list[dict]) -> None:
    """Save stimuli list to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STIMULI_FILE, "w") as f:
        json.dump(stimuli, f, indent=2)


def count_words(text: str) -> int:
    return len(text.split())


def generate_for_product(product: dict, variant: int) -> dict[str, str]:
    """Generate descriptions for all 5 registers for one product+variant.

    Returns dict mapping register -> description text.
    """
    # Build a combined prompt for all 5 registers at once
    prompt_parts = []
    prompt_parts.append(
        f"Generate 5 product descriptions for '{product['name']}' "
        f"(category: {product['category'].replace('_', ' ')}, "
        f"{'fictional' if product['is_fictional'] else 'real'} product).\n\n"
    )

    # Product info
    attr_lines = "\n".join(f"  - {k}: {v}" for k, v in product["core_attributes"].items())
    feat_lines = "\n".join(f"  - {f}" for f in product["distinguishing_features"])
    prompt_parts.append(f"Core Attributes:\n{attr_lines}\n")
    prompt_parts.append(f"Distinguishing Features:\n{feat_lines}\n\n")

    if variant == 0:
        diversity = "Write in a straightforward style for each register. Prioritize clarity and natural phrasing."
    else:
        diversity = (
            "Write a DIFFERENT version from your usual approach for each register. "
            "Vary sentence structure, word choice, and opening strategy compared to a standard version. "
            "Still respect register constraints but explore alternative phrasings and distinct rhetorical angles."
        )

    prompt_parts.append(f"Diversity instruction: {diversity}\n\n")

    prompt_parts.append("For each of the following 5 registers, write a product description that:\n")
    prompt_parts.append("- Is 80-150 words (HARD limits: 50-200 words)\n")
    prompt_parts.append("- Conveys ALL core attributes (numerical values, percentages, ingredients)\n")
    prompt_parts.append("- Does NOT use the product name as a heading/title\n")
    prompt_parts.append("- Contains NO meta-commentary\n")
    prompt_parts.append("- Is ONLY the description text, nothing else\n\n")

    prompt_parts.append("Output EXACTLY this JSON format (no markdown fences, no extra text):\n")
    prompt_parts.append("{\n")
    for i, reg in enumerate(REGISTERS):
        spec = REGISTER_SPECS[reg]
        prompt_parts.append(f'  "{reg}": "<description text>",\n')
        prompt_parts.append(f'  // {reg} register: voice={spec["voice"][:60]}..., tone={spec["tone"][:60]}...\n')
    prompt_parts.append("}\n\n")

    # Simplify: just ask for clean JSON
    prompt_parts = []
    prompt_parts.append(
        f"Generate 5 product descriptions for the product below, one per register.\n\n"
        f"PRODUCT: {product['name']}\n"
        f"Category: {product['category'].replace('_', ' ').title()}\n"
        f"Type: {'Fictional' if product['is_fictional'] else 'Real'}\n\n"
    )
    prompt_parts.append(f"Core Attributes:\n{attr_lines}\n\n")
    prompt_parts.append(f"Distinguishing Features:\n{feat_lines}\n\n")
    prompt_parts.append(f"Style note: {diversity}\n\n")
    prompt_parts.append("REGISTERS (write one description per register):\n\n")

    for reg in REGISTERS:
        spec = REGISTER_SPECS[reg]
        prompt_parts.append(f"**{reg}**:\n")
        prompt_parts.append(f"  Voice: {spec['voice']}\n")
        prompt_parts.append(f"  Tone: {spec['tone']}\n")
        prompt_parts.append(f"  Structure: {spec['structure']}\n")
        prompt_parts.append(f"  Vocabulary: {spec['vocabulary']}\n\n")

    prompt_parts.append(
        "CONSTRAINTS:\n"
        "- Each description: 80-150 words (hard limits: 50-200)\n"
        "- ALL core attributes must appear in the text (numbers, percentages, ingredients)\n"
        "- Do NOT use the product name as a heading\n"
        "- No meta-commentary (e.g., 'Here is...')\n"
        "- Output ONLY the descriptions\n\n"
    )
    prompt_parts.append(
        "Output valid JSON only — no markdown code fences, no commentary.\n"
        "Format:\n"
        '{"marketing": "...", "regulatory": "...", "casual_social": "...", "patent": "...", "journalistic": "..."}\n'
    )

    prompt = "".join(prompt_parts)

    # Call claude CLI
    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude CLI failed: {result.stderr}")

    # Parse JSON from output — strip any markdown fences
    output = result.stdout.strip()
    if output.startswith("```"):
        # Remove markdown fences
        lines = output.split("\n")
        # Find first { and last }
        start = next(i for i, l in enumerate(lines) if l.strip().startswith("{"))
        end = next(i for i in range(len(lines) - 1, -1, -1) if l.strip().startswith("}"))
        output = "\n".join(lines[start:end + 1])

    # More robust: find the JSON object
    brace_start = output.find("{")
    brace_end = output.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        output = output[brace_start:brace_end]

    try:
        descriptions = json.loads(output)
    except json.JSONDecodeError:
        # Try to fix common issues
        # Remove trailing commas
        import re
        cleaned = re.sub(r',\s*}', '}', output)
        cleaned = re.sub(r',\s*]', ']', cleaned)
        descriptions = json.loads(cleaned)

    return descriptions


def main():
    all_products = REAL_PRODUCTS + FICTIONAL_PRODUCTS
    existing = load_existing()
    existing_ids = {s["stimulus_id"] for s in existing}
    stimuli = list(existing)

    total_products = len(all_products)
    total_expected = total_products * len(REGISTERS) * 2  # 80 * 5 * 2 = 800

    print(f"Total products: {total_products}")
    print(f"Total stimuli expected: {total_expected}")
    print(f"Already generated: {len(existing)}")
    print()

    generated_count = 0
    skipped_count = 0
    error_count = 0

    for prod_idx, product in enumerate(all_products):
        for variant in range(2):
            # Check if all stimuli for this product+variant already exist
            expected_ids = {
                f"{product['id']}_{reg}_v{variant}" for reg in REGISTERS
            }
            if expected_ids.issubset(existing_ids):
                skipped_count += len(expected_ids)
                continue

            # Generate
            tag = f"[{prod_idx + 1}/{total_products}] {product['name']} v{variant}"
            print(f"{tag}: generating...", end=" ", flush=True)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    descriptions = generate_for_product(product, variant)

                    # Validate and build stimuli
                    for reg in REGISTERS:
                        stim_id = f"{product['id']}_{reg}_v{variant}"
                        if stim_id in existing_ids:
                            continue

                        text = descriptions.get(reg, "")
                        if not text:
                            print(f"\n  WARNING: empty {reg} description, retrying...")
                            raise ValueError(f"Empty description for {reg}")

                        word_count = count_words(text)
                        stimulus = {
                            "stimulus_id": stim_id,
                            "product_id": product["id"],
                            "category": product["category"],
                            "register": reg,
                            "variant": variant,
                            "is_fictional": product["is_fictional"],
                            "text": text,
                            "token_count": word_count,
                            "generator": "claude",
                            "core_attributes": product["core_attributes"],
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                        }
                        stimuli.append(stimulus)
                        existing_ids.add(stim_id)
                        generated_count += 1

                    print(f"OK ({generated_count} new)")
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"\n  Retry {attempt + 1}: {e}")
                        time.sleep(2)
                    else:
                        print(f"\n  FAILED after {max_retries} attempts: {e}")
                        error_count += 1

            # Save after each product+variant
            save_stimuli(stimuli)

    print(f"\nDone! Generated: {generated_count}, Skipped: {skipped_count}, Errors: {error_count}")
    print(f"Total stimuli: {len(stimuli)}")
    print(f"Saved to: {STIMULI_FILE}")


if __name__ == "__main__":
    main()
