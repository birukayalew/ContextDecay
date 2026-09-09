import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.extract.layer_select import classify_layers, select_cache_layers


def make_qwen_like_layer_types():
    # 64 layers, 16 full-attention interleaved among 48 linear-attention,
    # roughly every 4th layer full -- illustrative, not the real pattern.
    types = []
    for i in range(64):
        types.append("full_attention" if i % 4 == 3 else "linear_attention")
    assert types.count("full_attention") == 16
    return types


def test_classify_layers_reads_layer_types():
    hf_config = {"layer_types": ["full_attention", "linear_attention", "linear_attention", "full_attention"]}
    result = classify_layers(hf_config)
    assert result == ["full_attention", "linear_attention", "linear_attention", "full_attention"]


def test_classify_layers_missing_field_raises():
    with pytest.raises(KeyError):
        classify_layers({})


def test_classify_layers_recognizes_delta_variants():
    hf_config = {"layer_types": ["gated_deltanet", "full_attention"]}
    assert classify_layers(hf_config) == ["linear_attention", "full_attention"]


def test_select_cache_layers_guarantees_min_full_attention():
    layer_types = make_qwen_like_layer_types()
    choices = select_cache_layers(layer_types, depth_fractions=(0.25, 0.5, 0.75, 1.0), min_full_attention=2)
    assert len(choices) == 4
    n_full = sum(1 for c in choices if c.attn_type == "full_attention")
    assert n_full >= 2
    # no duplicate indices
    assert len(set(c.index for c in choices)) == 4


def test_select_cache_layers_all_full_attention_trivially_satisfies():
    layer_types = ["full_attention"] * 8
    choices = select_cache_layers(layer_types, depth_fractions=(0.25, 0.5, 0.75, 1.0), min_full_attention=2)
    assert all(c.attn_type == "full_attention" for c in choices)


def test_select_cache_layers_insufficient_full_layers_raises():
    layer_types = ["linear_attention"] * 10 + ["full_attention"]
    with pytest.raises(ValueError):
        select_cache_layers(layer_types, min_full_attention=2)
