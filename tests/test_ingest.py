"""Tests for coherence.ingest -- document ingestion pipeline."""

from __future__ import annotations

import json
import logging
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coherence.ingest import (
    CHANNELS,
    RealDocument,
    aggregate_reviews,
    clean_document,
    detect_multi_product,
    get_tokenizer,
    load_documents,
    to_stimuli_format,
    truncate_document,
    validate_product_set,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_doc():
    """A basic RealDocument for testing."""
    return RealDocument(
        product_id="oral_care_001",
        channel="regulatory",
        text="Stannous fluoride 0.454%. Anticavity toothpaste.",
    )


@pytest.fixture
def retail_doc_brand():
    return RealDocument(
        product_id="oral_care_001",
        channel="retail",
        text="Buy our amazing toothpaste today!",
        author="brand",
    )


@pytest.fixture
def retail_doc_third_party():
    return RealDocument(
        product_id="oral_care_001",
        channel="retail",
        text="Seller listing for Colgate Total.",
        author="third_party",
    )


@pytest.fixture
def retail_doc_no_author():
    return RealDocument(
        product_id="oral_care_001",
        channel="retail",
        text="Toothpaste product page.",
    )


@pytest.fixture
def portfolio_dir(tmp_path):
    """Create a minimal portfolio directory structure."""
    product_dir = tmp_path / "skincare_001"
    product_dir.mkdir()

    # Required channels
    (product_dir / "regulatory.txt").write_text(
        "Active ingredient: salicylic acid 2%.", encoding="utf-8"
    )
    (product_dir / "marketing.txt").write_text(
        "Transform your skin with our <b>revolutionary</b> formula!",
        encoding="utf-8",
    )
    (product_dir / "retail.txt").write_text(
        "Ships from and sold by Amazon. Great skincare product.",
        encoding="utf-8",
    )

    # Social channel
    (product_dir / "social.txt").write_text(
        "Love this product! #skincare @beautybrand Follow us for more tips!",
        encoding="utf-8",
    )

    # Consumer reviews
    review_dir = product_dir / "consumer_review"
    review_dir.mkdir()
    for i in range(12):
        word_count = 60 if i < 8 else 30  # 8 qualifying, 4 too short
        words = " ".join(["word"] * word_count)
        (review_dir / f"review_{i:03d}.txt").write_text(words, encoding="utf-8")

    # Metadata sidecar
    metadata = {
        "source_urls": {
            "regulatory": "https://fda.gov/skincare001",
            "marketing": "https://brand.com/skincare001",
        },
        "dates_collected": {
            "regulatory": "2025-01-15",
            "marketing": "2025-02-01",
        },
        "authors": {
            "retail": "brand",
        },
    }
    (product_dir / "metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    return tmp_path


@pytest.fixture
def mock_tokenizer():
    """A mock tokenizer that splits on whitespace for deterministic tests."""
    tok = MagicMock()

    def encode(text, **kwargs):
        # Simple whitespace tokenizer -- each word is one "token"
        return list(range(len(text.split())))

    def decode(token_ids, **kwargs):
        # Not a real decode, but good enough: we just take that many words
        # We store the original text for decode via a side channel
        return " ".join(["tok"] * len(token_ids))

    tok.encode = encode
    tok.decode = decode
    return tok


@pytest.fixture(autouse=True)
def _patch_tokenizer(mock_tokenizer):
    """Patch get_tokenizer globally so tests don't need the real model."""
    with patch("coherence.ingest.get_tokenizer", return_value=mock_tokenizer):
        yield


# ---------------------------------------------------------------------------
# RealDocument tests
# ---------------------------------------------------------------------------


class TestRealDocument:
    def test_creation_basic(self, sample_doc):
        assert sample_doc.product_id == "oral_care_001"
        assert sample_doc.channel == "regulatory"
        assert sample_doc.author is None
        assert sample_doc.truncated is False
        assert sample_doc.original_token_count is None
        assert sample_doc.multi_product_flag is False

    def test_invalid_channel_raises(self):
        with pytest.raises(ValueError, match="Invalid channel"):
            RealDocument(product_id="x", channel="invalid", text="hello")

    def test_invalid_author_raises(self):
        with pytest.raises(ValueError, match="Invalid author"):
            RealDocument(
                product_id="x", channel="regulatory", text="hello", author="unknown"
            )

    def test_valid_author_values(self):
        doc_brand = RealDocument(
            product_id="x", channel="retail", text="t", author="brand"
        )
        assert doc_brand.author == "brand"

        doc_tp = RealDocument(
            product_id="x", channel="retail", text="t", author="third_party"
        )
        assert doc_tp.author == "third_party"

    def test_frozen_immutability(self, sample_doc):
        with pytest.raises(AttributeError):
            sample_doc.text = "modified"  # type: ignore[misc]

    def test_default_field_values(self):
        doc = RealDocument(product_id="p", channel="marketing", text="t")
        assert doc.source_url == ""
        assert doc.date_collected == ""
        assert doc.author is None
        assert doc.truncated is False
        assert doc.original_token_count is None
        assert doc.multi_product_flag is False


# ---------------------------------------------------------------------------
# is_brand_controlled tests
# ---------------------------------------------------------------------------


class TestIsBrandControlled:
    def test_regulatory_always_true(self, sample_doc):
        assert sample_doc.is_brand_controlled is True

    def test_marketing_always_true(self):
        doc = RealDocument(product_id="x", channel="marketing", text="t")
        assert doc.is_brand_controlled is True

    def test_social_always_true(self):
        doc = RealDocument(product_id="x", channel="social", text="t")
        assert doc.is_brand_controlled is True

    def test_consumer_review_always_false(self):
        doc = RealDocument(product_id="x", channel="consumer_review", text="t")
        assert doc.is_brand_controlled is False

    def test_retail_brand_true(self, retail_doc_brand):
        assert retail_doc_brand.is_brand_controlled is True

    def test_retail_third_party_false(self, retail_doc_third_party):
        assert retail_doc_third_party.is_brand_controlled is False

    def test_retail_no_author_defaults_true_with_warning(
        self, retail_doc_no_author, caplog
    ):
        with caplog.at_level(logging.WARNING):
            result = retail_doc_no_author.is_brand_controlled
        assert result is True
        assert "no author" in caplog.text.lower()


# ---------------------------------------------------------------------------
# load_documents tests
# ---------------------------------------------------------------------------


class TestLoadDocuments:
    def test_loads_all_channels(self, portfolio_dir):
        docs = load_documents(portfolio_dir)
        channels = {d.channel for d in docs}
        assert "regulatory" in channels
        assert "marketing" in channels
        assert "retail" in channels
        assert "social" in channels
        assert "consumer_review" in channels

    def test_correct_document_count(self, portfolio_dir):
        docs = load_documents(portfolio_dir)
        # 4 top-level .txt + 12 reviews = 16
        assert len(docs) == 16

    def test_product_id_from_dirname(self, portfolio_dir):
        docs = load_documents(portfolio_dir)
        assert all(d.product_id == "skincare_001" for d in docs)

    def test_metadata_source_url(self, portfolio_dir):
        docs = load_documents(portfolio_dir)
        reg_doc = [d for d in docs if d.channel == "regulatory"][0]
        assert reg_doc.source_url == "https://fda.gov/skincare001"

    def test_metadata_date_collected(self, portfolio_dir):
        docs = load_documents(portfolio_dir)
        mkt_doc = [d for d in docs if d.channel == "marketing"][0]
        assert mkt_doc.date_collected == "2025-02-01"

    def test_metadata_author(self, portfolio_dir):
        docs = load_documents(portfolio_dir)
        retail_doc = [d for d in docs if d.channel == "retail"][0]
        assert retail_doc.author == "brand"

    def test_missing_metadata_defaults(self, portfolio_dir):
        docs = load_documents(portfolio_dir)
        social_doc = [d for d in docs if d.channel == "social"][0]
        # Social has no metadata entries
        assert social_doc.source_url == ""
        assert social_doc.date_collected == ""
        assert social_doc.author is None

    def test_no_metadata_file(self, tmp_path):
        """Portfolio without metadata.json still loads fine."""
        product_dir = tmp_path / "prod_001"
        product_dir.mkdir()
        (product_dir / "regulatory.txt").write_text("text", encoding="utf-8")
        (product_dir / "marketing.txt").write_text("text", encoding="utf-8")

        docs = load_documents(tmp_path)
        assert len(docs) == 2
        assert all(d.source_url == "" for d in docs)

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_documents(Path("/nonexistent/portfolio"))

    def test_skips_unknown_channel_files(self, tmp_path, caplog):
        product_dir = tmp_path / "prod_001"
        product_dir.mkdir()
        (product_dir / "regulatory.txt").write_text("text", encoding="utf-8")
        (product_dir / "unknown_channel.txt").write_text("skip me", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            docs = load_documents(tmp_path)
        assert len(docs) == 1
        assert "unknown channel" in caplog.text.lower()


# ---------------------------------------------------------------------------
# clean_document tests
# ---------------------------------------------------------------------------


class TestCleanDocument:
    def test_strips_html(self):
        doc = RealDocument(
            product_id="x",
            channel="marketing",
            text="This is <b>bold</b> and <a href='url'>linked</a>.",
        )
        cleaned = clean_document(doc)
        assert "<b>" not in cleaned.text
        assert "<a" not in cleaned.text
        assert "bold" in cleaned.text

    def test_removes_copyright(self):
        doc = RealDocument(
            product_id="x",
            channel="regulatory",
            text="Good product.\n\u00a9 2024 Company Inc. All rights reserved.\nMore text.",
        )
        cleaned = clean_document(doc)
        assert "\u00a9" not in cleaned.text
        assert "More text" in cleaned.text

    def test_removes_amazon_chrome(self):
        doc = RealDocument(
            product_id="x",
            channel="retail",
            text="Great product.\nCustomers who bought this item also bought\nOther stuff.",
        )
        cleaned = clean_document(doc)
        assert "Customers who bought" not in cleaned.text

    def test_removes_hashtags_mentions(self):
        doc = RealDocument(
            product_id="x",
            channel="social",
            text="Love this! #skincare @brand Great product.",
        )
        cleaned = clean_document(doc)
        assert "#skincare" not in cleaned.text
        assert "@brand" not in cleaned.text
        assert "Love this!" in cleaned.text

    def test_normalizes_unicode(self):
        # e + combining acute = NFC should produce single char
        doc = RealDocument(
            product_id="x",
            channel="regulatory",
            text="cafe\u0301",
        )
        cleaned = clean_document(doc)
        assert cleaned.text == "caf\u00e9"

    def test_does_not_mutate_input(self):
        doc = RealDocument(
            product_id="x",
            channel="marketing",
            text="Some <b>HTML</b> text.",
        )
        original_text = doc.text
        cleaned = clean_document(doc)
        assert doc.text == original_text
        assert cleaned is not doc

    def test_channel_specific_boilerplate_regulatory(self):
        doc = RealDocument(
            product_id="x",
            channel="regulatory",
            text="Active ingredient. These statements have not been evaluated by the Food and Drug Administration. More info.",
        )
        cleaned = clean_document(doc)
        assert "These statements have not been evaluated" not in cleaned.text
        assert "Active ingredient" in cleaned.text

    def test_channel_specific_boilerplate_retail(self):
        doc = RealDocument(
            product_id="x",
            channel="retail",
            text="Great product. Ships from and sold by Amazon. Buy it now.",
        )
        cleaned = clean_document(doc)
        assert "Ships from and sold by" not in cleaned.text

    def test_collapses_whitespace(self):
        doc = RealDocument(
            product_id="x",
            channel="marketing",
            text="Line 1.\n\n\n\n\nLine 2.   Extra   spaces.",
        )
        cleaned = clean_document(doc)
        assert "\n\n\n" not in cleaned.text
        assert "  " not in cleaned.text


# ---------------------------------------------------------------------------
# truncate_document tests
# ---------------------------------------------------------------------------


class TestTruncateDocument:
    def test_within_limit_unchanged(self, mock_tokenizer):
        doc = RealDocument(
            product_id="x",
            channel="regulatory",
            text="short text here",
        )
        result = truncate_document(doc, max_tokens=100)
        assert result is doc  # unchanged means same object
        assert result.truncated is False
        assert result.original_token_count is None

    def test_exceeds_limit_truncated(self, mock_tokenizer):
        # 20 words -> 20 "tokens" with our mock
        long_text = " ".join(["word"] * 20)
        doc = RealDocument(product_id="x", channel="regulatory", text=long_text)
        result = truncate_document(doc, max_tokens=5)
        assert result.truncated is True
        assert result.original_token_count == 20

    def test_does_not_mutate_input(self, mock_tokenizer):
        long_text = " ".join(["word"] * 20)
        doc = RealDocument(product_id="x", channel="regulatory", text=long_text)
        original_text = doc.text
        truncate_document(doc, max_tokens=5)
        assert doc.text == original_text

    def test_logs_warning_on_truncation(self, mock_tokenizer, caplog):
        long_text = " ".join(["word"] * 100)
        doc = RealDocument(product_id="x", channel="marketing", text=long_text)
        with caplog.at_level(logging.WARNING):
            truncate_document(doc, max_tokens=10)
        assert "truncating" in caplog.text.lower()


# ---------------------------------------------------------------------------
# to_stimuli_format tests
# ---------------------------------------------------------------------------


class TestToStimuliFormat:
    def test_returns_dict_with_required_keys(self, sample_doc):
        result = to_stimuli_format(sample_doc)
        assert "stimulus_id" in result
        assert "text" in result
        assert isinstance(result["stimulus_id"], str)
        assert isinstance(result["text"], str)

    def test_stimulus_id_format(self, sample_doc):
        result = to_stimuli_format(sample_doc, index=0)
        assert result["stimulus_id"] == "oral_care_001_regulatory_0"

    def test_stimulus_id_with_index(self):
        doc = RealDocument(
            product_id="skincare_001",
            channel="consumer_review",
            text="Great product!",
        )
        result = to_stimuli_format(doc, index=3)
        assert result["stimulus_id"] == "skincare_001_consumer_review_3"

    def test_text_field_matches_doc(self, sample_doc):
        result = to_stimuli_format(sample_doc)
        assert result["text"] == sample_doc.text

    def test_metadata_fields_present(self, sample_doc):
        result = to_stimuli_format(sample_doc)
        assert "product_id" in result
        assert "channel" in result
        assert "date_collected" in result

    def test_compatible_with_extraction(self, sample_doc):
        """Verify the output has the keys extraction.py accesses."""
        result = to_stimuli_format(sample_doc)
        # extraction.py accesses stim["stimulus_id"] and stim["text"]
        _ = result["stimulus_id"]
        _ = result["text"]


# ---------------------------------------------------------------------------
# validate_product_set tests
# ---------------------------------------------------------------------------


class TestValidateProductSet:
    def test_valid_product(self):
        docs = [
            RealDocument(product_id="p1", channel="regulatory", text="t"),
            RealDocument(product_id="p1", channel="marketing", text="t"),
            RealDocument(product_id="p1", channel="retail", text="t", author="brand"),
        ]
        result = validate_product_set(docs)
        assert result["p1"]["valid"] is True
        assert len(result["p1"]["channels_missing"]) == 0

    def test_missing_required_channel(self, caplog):
        docs = [
            RealDocument(product_id="p1", channel="regulatory", text="t"),
            RealDocument(product_id="p1", channel="retail", text="t", author="brand"),
            RealDocument(product_id="p1", channel="social", text="t"),
        ]
        with caplog.at_level(logging.WARNING):
            result = validate_product_set(docs)
        assert result["p1"]["valid"] is False
        assert "marketing" in result["p1"]["channels_missing"]
        assert "fails validation" in caplog.text.lower()

    def test_too_few_channels(self, caplog):
        docs = [
            RealDocument(product_id="p1", channel="regulatory", text="t"),
            RealDocument(product_id="p1", channel="marketing", text="t"),
        ]
        with caplog.at_level(logging.WARNING):
            result = validate_product_set(docs)
        assert result["p1"]["valid"] is False

    def test_no_optional_channel(self, caplog):
        """Has regulatory + marketing but needs at least one optional."""
        docs = [
            RealDocument(product_id="p1", channel="regulatory", text="t"),
            RealDocument(product_id="p1", channel="marketing", text="t"),
        ]
        with caplog.at_level(logging.WARNING):
            result = validate_product_set(docs)
        assert result["p1"]["valid"] is False

    def test_multiple_products(self):
        docs = [
            RealDocument(product_id="p1", channel="regulatory", text="t"),
            RealDocument(product_id="p1", channel="marketing", text="t"),
            RealDocument(product_id="p1", channel="retail", text="t", author="brand"),
            RealDocument(product_id="p2", channel="regulatory", text="t"),
            RealDocument(product_id="p2", channel="marketing", text="t"),
            RealDocument(product_id="p2", channel="consumer_review", text="t"),
        ]
        result = validate_product_set(docs)
        assert result["p1"]["valid"] is True
        assert result["p2"]["valid"] is True

    def test_channels_present_list(self):
        docs = [
            RealDocument(product_id="p1", channel="regulatory", text="t"),
            RealDocument(product_id="p1", channel="marketing", text="t"),
            RealDocument(product_id="p1", channel="social", text="t"),
        ]
        result = validate_product_set(docs)
        assert sorted(result["p1"]["channels_present"]) == [
            "marketing",
            "regulatory",
            "social",
        ]

    def test_does_not_raise(self, caplog):
        """validate_product_set should log warnings, never raise."""
        docs = [
            RealDocument(product_id="p1", channel="regulatory", text="t"),
        ]
        with caplog.at_level(logging.WARNING):
            result = validate_product_set(docs)
        assert result["p1"]["valid"] is False
        # No exception raised


# ---------------------------------------------------------------------------
# aggregate_reviews tests
# ---------------------------------------------------------------------------


class TestAggregateReviews:
    def _make_reviews(self, product_id, count, word_count=60):
        return [
            RealDocument(
                product_id=product_id,
                channel="consumer_review",
                text=" ".join(["word"] * word_count),
            )
            for _ in range(count)
        ]

    def test_returns_up_to_10(self):
        reviews = self._make_reviews("p1", 15)
        result = aggregate_reviews(reviews, "p1")
        assert len(result) == 10

    def test_filters_by_product_id(self):
        reviews_p1 = self._make_reviews("p1", 5)
        reviews_p2 = self._make_reviews("p2", 5)
        result = aggregate_reviews(reviews_p1 + reviews_p2, "p1")
        assert all(r.product_id == "p1" for r in result)

    def test_filters_by_channel(self):
        reviews = self._make_reviews("p1", 5)
        non_review = RealDocument(product_id="p1", channel="marketing", text="word " * 60)
        result = aggregate_reviews(reviews + [non_review], "p1")
        assert all(r.channel == "consumer_review" for r in result)

    def test_filters_short_reviews(self):
        long_reviews = self._make_reviews("p1", 3, word_count=60)
        short_reviews = self._make_reviews("p1", 3, word_count=30)
        result = aggregate_reviews(long_reviews + short_reviews, "p1")
        assert len(result) == 3

    def test_warns_when_fewer_than_10(self, caplog):
        reviews = self._make_reviews("p1", 5)
        with caplog.at_level(logging.WARNING):
            result = aggregate_reviews(reviews, "p1")
        assert len(result) == 5
        assert "5 qualifying reviews" in caplog.text

    def test_exactly_50_words_qualifies(self):
        review = RealDocument(
            product_id="p1",
            channel="consumer_review",
            text=" ".join(["word"] * 50),
        )
        result = aggregate_reviews([review], "p1")
        assert len(result) == 1

    def test_49_words_excluded(self):
        review = RealDocument(
            product_id="p1",
            channel="consumer_review",
            text=" ".join(["word"] * 49),
        )
        result = aggregate_reviews([review], "p1")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# detect_multi_product tests
# ---------------------------------------------------------------------------


class TestDetectMultiProduct:
    def test_no_cross_mention(self):
        doc = RealDocument(
            product_id="oral_care_001",
            channel="regulatory",
            text="Our toothpaste is great.",
        )
        product_names = {
            "oral_care_001": "Colgate Total",
            "oral_care_002": "Crest Pro-Health",
        }
        result = detect_multi_product(doc, product_names)
        assert result.multi_product_flag is False

    def test_detects_cross_mention(self):
        doc = RealDocument(
            product_id="oral_care_001",
            channel="retail",
            text="Better than Crest Pro-Health in every way!",
            author="brand",
        )
        product_names = {
            "oral_care_001": "Colgate Total",
            "oral_care_002": "Crest Pro-Health",
        }
        result = detect_multi_product(doc, product_names)
        assert result.multi_product_flag is True

    def test_case_insensitive(self):
        doc = RealDocument(
            product_id="oral_care_001",
            channel="retail",
            text="Works better than crest pro-health.",
            author="brand",
        )
        product_names = {
            "oral_care_001": "Colgate Total",
            "oral_care_002": "Crest Pro-Health",
        }
        result = detect_multi_product(doc, product_names)
        assert result.multi_product_flag is True

    def test_own_product_not_flagged(self):
        doc = RealDocument(
            product_id="oral_care_001",
            channel="marketing",
            text="Colgate Total is the best toothpaste ever!",
        )
        product_names = {
            "oral_care_001": "Colgate Total",
            "oral_care_002": "Crest Pro-Health",
        }
        result = detect_multi_product(doc, product_names)
        assert result.multi_product_flag is False

    def test_missing_product_id_raises(self):
        doc = RealDocument(
            product_id="unknown_001",
            channel="regulatory",
            text="Some text.",
        )
        product_names = {
            "oral_care_001": "Colgate Total",
        }
        with pytest.raises(KeyError, match="unknown_001"):
            detect_multi_product(doc, product_names)

    def test_does_not_mutate_input(self):
        doc = RealDocument(
            product_id="oral_care_001",
            channel="retail",
            text="Better than Crest Pro-Health!",
            author="brand",
        )
        product_names = {
            "oral_care_001": "Colgate Total",
            "oral_care_002": "Crest Pro-Health",
        }
        original_flag = doc.multi_product_flag
        detect_multi_product(doc, product_names)
        assert doc.multi_product_flag == original_flag

    def test_logs_warning_on_detection(self, caplog):
        doc = RealDocument(
            product_id="oral_care_001",
            channel="retail",
            text="Better than Crest Pro-Health!",
            author="brand",
        )
        product_names = {
            "oral_care_001": "Colgate Total",
            "oral_care_002": "Crest Pro-Health",
        }
        with caplog.at_level(logging.WARNING):
            detect_multi_product(doc, product_names)
        assert "mentions other product" in caplog.text.lower()


# ---------------------------------------------------------------------------
# Integration: load + clean + validate pipeline
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def test_load_clean_validate(self, portfolio_dir):
        # Load
        docs = load_documents(portfolio_dir)
        assert len(docs) > 0

        # Clean
        cleaned = [clean_document(d) for d in docs]
        assert all(isinstance(d, RealDocument) for d in cleaned)

        # Validate
        result = validate_product_set(cleaned)
        assert "skincare_001" in result

    def test_to_stimuli_format_all_docs(self, portfolio_dir):
        docs = load_documents(portfolio_dir)
        stimuli = []
        review_idx: dict[str, int] = {}
        for doc in docs:
            if doc.channel == "consumer_review":
                key = doc.product_id
                idx = review_idx.get(key, 0)
                review_idx[key] = idx + 1
            else:
                idx = 0
            stimuli.append(to_stimuli_format(doc, index=idx))

        # Every stimulus has the extraction-required keys
        for s in stimuli:
            assert "stimulus_id" in s
            assert "text" in s

        # All stimulus_ids are unique
        ids = [s["stimulus_id"] for s in stimuli]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Channel taxonomy
# ---------------------------------------------------------------------------


class TestChannelTaxonomy:
    def test_expected_channels(self):
        assert CHANNELS == {
            "regulatory",
            "marketing",
            "retail",
            "social",
            "consumer_review",
        }
