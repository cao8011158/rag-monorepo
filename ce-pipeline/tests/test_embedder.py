from __future__ import annotations

import numpy as np
import pytest

from ce_pipeline.embedding.embedder import DualInstructEmbedder


class FakeSentenceTransformer:
    """
    Fake SentenceTransformer to avoid downloading real models in tests.

    - Records init args
    - Records encode() inputs/kwargs
    - Returns deterministic numpy arrays (float64 on purpose to test float32 cast)
    """

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self.last_encode_inputs: list[str] | None = None
        self.last_encode_kwargs: dict | None = None

    def encode(
        self,
        inputs,
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ):
        self.last_encode_inputs = list(inputs)
        self.last_encode_kwargs = {
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
            "convert_to_numpy": convert_to_numpy,
            "normalize_embeddings": normalize_embeddings,
        }

        # Return float64 to verify that embedder casts to float32
        n = len(self.last_encode_inputs)
        d = 3
        return np.arange(n * d, dtype=np.float64).reshape(n, d)


def patch_sentence_transformer(monkeypatch) -> None:
    """
    Patch SentenceTransformer symbol inside ce_pipeline.embedding.embedder module.
    """
    import ce_pipeline.embedding.embedder as mod

    monkeypatch.setattr(mod, "SentenceTransformer", FakeSentenceTransformer)


def test_init_with_device(monkeypatch):
    patch_sentence_transformer(monkeypatch)

    emb = DualInstructEmbedder(
        model_name="dummy/model",
        passage_instruction="passage: ",
        query_instruction="query: ",
        device="cpu",
    )

    assert emb._model.model_name == "dummy/model"
    assert emb._model.device == "cpu"


def test_init_without_device(monkeypatch):
    patch_sentence_transformer(monkeypatch)

    emb = DualInstructEmbedder(
        model_name="dummy/model",
        passage_instruction="passage: ",
        query_instruction="query: ",
        device=None,
    )

    assert emb._model.model_name == "dummy/model"
    assert emb._model.device is None


def test_encode_passages_uses_passage_instruction_and_casts_float32(monkeypatch):
    patch_sentence_transformer(monkeypatch)

    emb = DualInstructEmbedder(
        model_name="dummy/model",
        passage_instruction="passage: ",
        query_instruction="query: ",
        batch_size=7,
        normalize_embeddings=True,
        device=None,
    )

    vec = emb.encode_passages(["A", "B"])

    # instruction prefix + newline
    assert emb._model.last_encode_inputs == ["passage: \nA", "passage: \nB"]

    # kwargs forwarded correctly
    assert emb._model.last_encode_kwargs == {
        "batch_size": 7,
        "show_progress_bar": False,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (2, 3)
    assert vec.dtype == np.float32


def test_encode_queries_uses_query_instruction(monkeypatch):
    patch_sentence_transformer(monkeypatch)

    emb = DualInstructEmbedder(
        model_name="dummy/model",
        passage_instruction="passage: ",
        query_instruction="query: ",
    )

    vec = emb.encode_queries(["Q1"])

    assert emb._model.last_encode_inputs == ["query: \nQ1"]
    assert vec.shape == (1, 3)
    assert vec.dtype == np.float32


def test_empty_inputs_return_empty_matrix(monkeypatch):
    patch_sentence_transformer(monkeypatch)

    emb = DualInstructEmbedder(
        model_name="dummy/model",
        passage_instruction="passage: ",
        query_instruction="query: ",
    )

    v1 = emb.encode_passages([])
    v2 = emb.encode_queries([])

    assert isinstance(v1, np.ndarray)
    assert isinstance(v2, np.ndarray)
    assert v1.shape == (0, 0)
    assert v2.shape == (0, 0)
    assert v1.dtype == np.float32
    assert v2.dtype == np.float32


def test_model_load_error_is_wrapped(monkeypatch):
    import ce_pipeline.embedding.embedder as mod

    class Boom:
        def __init__(self, *args, **kwargs):
            raise OSError("no model file")

    monkeypatch.setattr(mod, "SentenceTransformer", Boom)

    with pytest.raises(RuntimeError) as e:
        DualInstructEmbedder(
            model_name="dummy/model",
            passage_instruction="passage: ",
            query_instruction="query: ",
        )

    msg = str(e.value)
    assert "Failed to load embedding model 'dummy/model'" in msg
    assert "no model file" in msg
