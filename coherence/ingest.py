"""Document ingestion for coherence analysis.

Loads, cleans, validates, and formats real-product documents for downstream
hidden-state extraction.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel taxonomy
# ---------------------------------------------------------------------------

CHANNELS = frozenset({
    "regulatory",
    "marketing",
    "retail",
    "social",
    "consumer_review",
})

# ---------------------------------------------------------------------------
# Tokenizer (lazy singleton)
# ---------------------------------------------------------------------------

_TOKENIZER_MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"
_tokenizer_instance = None


def get_tokenizer():
    """Return the shared tokenizer instance (lazy-loaded singleton)."""
    global _tokenizer_instance  # noqa: PLW0603
    if _tokenizer_instance is None:
        from transformers import AutoTokenizer

        _tokenizer_instance = AutoTokenizer.from_pretrained(
            _TOKENIZER_MODEL, trust_remote_code=True
        )
        logger.info("Loaded tokenizer: %s", _TOKENIZER_MODEL)
    return _tokenizer_instance


# ---------------------------------------------------------------------------
# RealDocument dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealDocument:
    """A single real-product document for coherence analysis."""

    product_id: str
    channel: str
    text: str
    source_url: str = ""
    date_collected: str = ""
    author: Optional[str] = None  # "brand" or "third_party"
    truncated: bool = False
    original_token_count: Optional[int] = None
    multi_product_flag: bool = False

    def __post_init__(self):
        if self.channel not in CHANNELS:
            raise ValueError(
                f"Invalid channel {self.channel!r}; must be one of {sorted(CHANNELS)}"
            )
        if self.author is not None and self.author not in ("brand", "third_party"):
            raise ValueError(
                f"Invalid author {self.author!r}; must be 'brand', 'third_party', or None"
            )

    @property
    def is_brand_controlled(self) -> bool:
        """Whether this document is brand-controlled, derived from channel and author."""
        if self.channel == "regulatory":
            return True
        if self.channel == "marketing":
            return True
        if self.channel == "social":
            return True
        if self.channel == "consumer_review":
            return False
        if self.channel == "retail":
            if self.author == "brand":
                return True
            if self.author == "third_party":
                return False
            # author is None -- default to True with warning
            logger.warning(
                "Retail document for %s has no author; defaulting is_brand_controlled=True",
                self.product_id,
            )
            return True
        # Should not reach here given __post_init__ validation
        raise ValueError(f"Unhandled channel: {self.channel!r}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_documents(portfolio_dir: Path) -> list[RealDocument]:
    """Load all documents from a portfolio directory.

    Expected structure::

        portfolio_dir/
          {product_id}/
            regulatory.txt
            marketing.txt
            retail.txt          (optional)
            social.txt          (optional)
            consumer_review/
              review_001.txt
              review_002.txt
              ...
            metadata.json       (optional)
    """
    portfolio_dir = Path(portfolio_dir)
    if not portfolio_dir.is_dir():
        raise FileNotFoundError(f"Portfolio directory not found: {portfolio_dir}")

    documents: list[RealDocument] = []

    for product_dir in sorted(portfolio_dir.iterdir()):
        if not product_dir.is_dir():
            continue

        product_id = product_dir.name

        # Load optional metadata sidecar
        metadata: dict = {}
        meta_path = product_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)

        source_urls: dict = metadata.get("source_urls", {})
        dates_collected: dict = metadata.get("dates_collected", {})
        authors: dict = metadata.get("authors", {})

        # Process top-level .txt files (excluding consumer_review subdir)
        for txt_file in sorted(product_dir.glob("*.txt")):
            channel = txt_file.stem
            if channel not in CHANNELS:
                logger.warning("Skipping unknown channel file: %s", txt_file)
                continue

            text = txt_file.read_text(encoding="utf-8")
            documents.append(
                RealDocument(
                    product_id=product_id,
                    channel=channel,
                    text=text,
                    source_url=source_urls.get(channel, ""),
                    date_collected=dates_collected.get(channel, ""),
                    author=authors.get(channel, None),
                )
            )

        # Process consumer_review subdirectory
        review_dir = product_dir / "consumer_review"
        if review_dir.is_dir():
            for review_file in sorted(review_dir.glob("*.txt")):
                text = review_file.read_text(encoding="utf-8")
                documents.append(
                    RealDocument(
                        product_id=product_id,
                        channel="consumer_review",
                        text=text,
                        source_url=source_urls.get("consumer_review", ""),
                        date_collected=dates_collected.get("consumer_review", ""),
                        author=authors.get("consumer_review", None),
                    )
                )

    logger.info("Loaded %d documents from %s", len(documents), portfolio_dir)
    return documents


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

# Pre-compiled patterns for cleaning
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_COPYRIGHT_RE = re.compile(
    r"(?:^|\n)\s*(?:\u00a9|Copyright|\(c\)).*?(?:\n|$)", re.IGNORECASE
)
_AMAZON_CHROME_RE = re.compile(
    r"(?:Customers who bought this item also bought|"
    r"Frequently bought together|"
    r"Customers who viewed this item also viewed|"
    r"Sponsored products related to this item|"
    r"Have a question\?|"
    r"See questions and answers|"
    r"Customer Questions & Answers|"
    r"Pages with related products|"
    r"Back to top|"
    r"Get to Know Us|"
    r"Make Money with Us|"
    r"Amazon Payment Products|"
    r"Let Us Help You).*?(?:\n|$)",
    re.IGNORECASE,
)
_HASHTAG_MENTION_RE = re.compile(r"(?:#\w+|@\w+)")

# Channel-specific boilerplate patterns
_REGULATORY_BOILERPLATE_RE = re.compile(
    r"(?:These statements have not been evaluated by the Food and Drug Administration|"
    r"This product is not intended to diagnose, treat, cure, or prevent any disease|"
    r"Consult your doctor before use|"
    r"Keep out of reach of children)\.?",
    re.IGNORECASE,
)
_MARKETING_BOILERPLATE_RE = re.compile(
    r"(?:Results may vary|"
    r"Individual results may vary|"
    r"\*Based on a survey of \d+ (?:users|customers|respondents)|"
    r"Terms and conditions apply|"
    r"Limited time offer|"
    r"While supplies last)\.?",
    re.IGNORECASE,
)
_RETAIL_BOILERPLATE_RE = re.compile(
    r"(?:Add to Cart|Add to List|"
    r"Buy Now|"
    r"Ships from and sold by|"
    r"Fulfilled by Amazon|"
    r"FREE (?:Shipping|delivery)|"
    r"In Stock|"
    r"Only \d+ left in stock|"
    r"Prime FREE Delivery|"
    r"\d+ out of 5 stars)\.?",
    re.IGNORECASE,
)
_SOCIAL_BOILERPLATE_RE = re.compile(
    r"(?:Follow us|"
    r"Like and share|"
    r"Tag a friend|"
    r"Link in bio|"
    r"Shop now|"
    r"Swipe up|"
    r"DM us for|"
    r"Click the link).*?(?:\n|$)",
    re.IGNORECASE,
)

_CHANNEL_BOILERPLATE = {
    "regulatory": _REGULATORY_BOILERPLATE_RE,
    "marketing": _MARKETING_BOILERPLATE_RE,
    "retail": _RETAIL_BOILERPLATE_RE,
    "social": _SOCIAL_BOILERPLATE_RE,
    "consumer_review": _AMAZON_CHROME_RE,
}


def clean_document(doc: RealDocument) -> RealDocument:
    """Clean a document: strip HTML, boilerplate, normalize Unicode.

    Returns a new RealDocument with cleaned text; does not mutate the input.
    """
    text = doc.text

    # 1. Strip HTML tags
    text = _HTML_TAG_RE.sub("", text)

    # 2. Remove copyright notices
    text = _COPYRIGHT_RE.sub("\n", text)

    # 3. Remove Amazon template chrome
    text = _AMAZON_CHROME_RE.sub("", text)

    # 4. Remove hashtags and mentions
    text = _HASHTAG_MENTION_RE.sub("", text)

    # 5. Channel-specific boilerplate removal
    channel_re = _CHANNEL_BOILERPLATE.get(doc.channel)
    if channel_re is not None:
        text = channel_re.sub("", text)

    # 6. Normalize Unicode (NFC)
    text = unicodedata.normalize("NFC", text)

    # 7. Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return replace(doc, text=text)


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------


def truncate_document(doc: RealDocument, max_tokens: int = 8192) -> RealDocument:
    """Truncate document text to max_tokens using the Qwen2.5-32B tokenizer.

    If the document exceeds max_tokens, returns a new RealDocument with truncated
    text, truncated=True, and original_token_count set. Logs a warning.
    If within the limit, returns unchanged.
    """
    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(doc.text)
    token_count = len(token_ids)

    if token_count <= max_tokens:
        return doc

    logger.warning(
        "Truncating document %s/%s from %d to %d tokens",
        doc.product_id,
        doc.channel,
        token_count,
        max_tokens,
    )

    truncated_text = tokenizer.decode(
        token_ids[:max_tokens], skip_special_tokens=True
    )

    return replace(
        doc,
        text=truncated_text,
        truncated=True,
        original_token_count=token_count,
    )


def normalize_document_length(
    docs: list[RealDocument], target_tokens: int = 512
) -> list[RealDocument]:
    """Normalize all documents to a common token length.

    Truncates documents longer than target_tokens and pads short documents by
    repeating their text until they reach the target. This ensures mean-pooled
    embeddings are computed over similar-length sequences, removing document
    length as a confound in coherence scores.

    Returns new RealDocument objects; does not mutate inputs.
    """
    tokenizer = get_tokenizer()
    result = []

    for doc in docs:
        token_ids = tokenizer.encode(doc.text)
        original_count = len(token_ids)

        if original_count == target_tokens:
            result.append(doc)
            continue

        if original_count > target_tokens:
            # Truncate
            normalized_text = tokenizer.decode(
                token_ids[:target_tokens], skip_special_tokens=True
            )
        else:
            # Pad by repeating text until we reach target length
            repeated_ids = token_ids.copy()
            while len(repeated_ids) < target_tokens:
                repeated_ids.extend(token_ids)
            normalized_text = tokenizer.decode(
                repeated_ids[:target_tokens], skip_special_tokens=True
            )

        result.append(
            replace(
                doc,
                text=normalized_text,
                truncated=original_count > target_tokens,
                original_token_count=original_count,
            )
        )

    logger.info(
        "Normalized %d documents to %d tokens each",
        len(result),
        target_tokens,
    )
    return result


# ---------------------------------------------------------------------------
# Stimuli format conversion
# ---------------------------------------------------------------------------


def to_stimuli_format(doc: RealDocument, index: int = 0) -> dict:
    """Convert a RealDocument to the dict format expected by extraction.py.

    Returns a dict with keys:
        - stimulus_id: "{product_id}_{channel}_{index}"
        - text: the document text
        - product_id, channel, date_collected: metadata for traceability
    """
    stimulus_id = f"{doc.product_id}_{doc.channel}_{index}"
    return {
        "stimulus_id": stimulus_id,
        "text": doc.text,
        "product_id": doc.product_id,
        "channel": doc.channel,
        "date_collected": doc.date_collected,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_REQUIRED_CHANNELS = {"regulatory", "marketing"}
_OPTIONAL_CHANNELS = {"retail", "social", "consumer_review"}


def validate_product_set(documents: list[RealDocument]) -> dict[str, dict]:
    """Verify minimum channel coverage per product.

    Requirements:
        - Minimum 3 channels per product
        - regulatory AND marketing must both be present
        - At least one of: retail, social, consumer_review

    Returns {product_id: {"valid": bool, "channels_present": list, "channels_missing": list}}.
    Logs a warning (does NOT raise) for products that fail validation.
    """
    # Group by product_id
    products: dict[str, set[str]] = {}
    for doc in documents:
        products.setdefault(doc.product_id, set()).add(doc.channel)

    results: dict[str, dict] = {}
    for product_id, channels_present in sorted(products.items()):
        missing_required = _REQUIRED_CHANNELS - channels_present
        has_optional = bool(channels_present & _OPTIONAL_CHANNELS)
        valid = (
            len(channels_present) >= 3
            and len(missing_required) == 0
            and has_optional
        )

        # Compute channels_missing: required channels that are absent +
        # note if no optional channels exist
        channels_missing = sorted(missing_required)
        if not has_optional:
            channels_missing.append("at_least_one_of(retail|social|consumer_review)")

        if not valid:
            logger.warning(
                "Product %s fails validation: present=%s, missing=%s",
                product_id,
                sorted(channels_present),
                channels_missing,
            )

        results[product_id] = {
            "valid": valid,
            "channels_present": sorted(channels_present),
            "channels_missing": channels_missing,
        }

    return results


# ---------------------------------------------------------------------------
# Review aggregation
# ---------------------------------------------------------------------------


def aggregate_reviews(
    reviews: list[RealDocument], product_id: str
) -> list[RealDocument]:
    """Filter and aggregate consumer reviews for a product.

    Filters to reviews matching the given product_id and channel="consumer_review",
    keeps only those with >= 50 words, returns up to 10.
    """
    matching = [
        r
        for r in reviews
        if r.product_id == product_id and r.channel == "consumer_review"
    ]

    qualifying = [r for r in matching if len(r.text.split()) >= 50]

    if len(qualifying) < 10:
        logger.warning(
            "Product %s has only %d qualifying reviews (>= 50 words); returning all",
            product_id,
            len(qualifying),
        )

    return qualifying[:10]


# ---------------------------------------------------------------------------
# Multi-product detection
# ---------------------------------------------------------------------------


def detect_multi_product(
    doc: RealDocument, product_names: dict[str, str]
) -> RealDocument:
    """Check whether a document mentions products other than its own.

    Args:
        doc: The document to check.
        product_names: Maps product_id -> product_name.

    Returns:
        A new RealDocument with multi_product_flag=True if another product
        is mentioned. Raises KeyError if doc.product_id not in product_names.
    """
    if doc.product_id not in product_names:
        raise KeyError(
            f"product_id {doc.product_id!r} not found in product_names"
        )

    own_name = product_names[doc.product_id]
    text_lower = doc.text.lower()

    for pid, pname in product_names.items():
        if pid == doc.product_id:
            continue
        if pname.lower() in text_lower:
            logger.warning(
                "Document %s/%s mentions other product %r (%s)",
                doc.product_id,
                doc.channel,
                pname,
                pid,
            )
            return replace(doc, multi_product_flag=True)

    return doc
