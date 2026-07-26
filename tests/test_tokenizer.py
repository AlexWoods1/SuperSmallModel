"""Tests for the byte-level BPE tokenizer."""

from __future__ import annotations

import pytest

from stlm.tokenizer import Tokenizer


def test_vocab_size_base_and_after_train() -> None:
    tok = Tokenizer()
    assert tok.vocab_size == 256
    tok.train("hello hello hello world", vocab_size=260)
    assert tok.vocab_size == 260
    assert len(tok.merges) == 4


def test_train_rejects_vocab_below_256() -> None:
    tok = Tokenizer()
    with pytest.raises(ValueError, match="greater than 256"):
        tok.train("abc", vocab_size=255)


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "Hello, world!\n",
        "café 日本語",
        "a" * 50,
    ],
)
def test_encode_decode_roundtrip(text: str) -> None:
    tok = Tokenizer()
    tok.train(text * 3, vocab_size=280)
    assert tok.decode(tok.encode(text)) == text


def test_encode_without_merges_is_utf8_bytes() -> None:
    tok = Tokenizer()
    text = "Hi"
    assert tok.encode(text) == list(text.encode("utf-8"))


def test_save_and_load_preserves_merges(tmp_path) -> None:
    tok = Tokenizer()
    tok.train("banana bandana banana", vocab_size=270)
    path = tmp_path / "tokenizer.json"
    tok.save(path)

    loaded = Tokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("banana") == tok.encode("banana")


def test_merge_replaces_adjacent_pair() -> None:
    ids = [1, 2, 1, 2, 3]
    assert Tokenizer._merge(ids, (1, 2), 99) == [99, 99, 3]


def test_count_pairs() -> None:
    counts = Tokenizer._count_pairs([1, 2, 1, 2])
    assert counts[(1, 2)] == 2
    assert counts[(2, 1)] == 1
