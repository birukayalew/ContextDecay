"""Layer selection for activation caching (protocol §1.3), adapted for
Qwen3.8-27B's hybrid attention architecture.

Qwen3.8-27B is NOT uniform softmax attention: 48 Gated DeltaNet
(linear-attention) layers vs. 16 full Gated Attention layers, interleaved.
Blindly taking depth fractions [0.25, 0.5, 0.75, 1.0] of 64 layers can land
on linear-attention layers, which do not do the same normalized all-to-all
softmax competition the attention-dilution story (protocol §7.1) depends
on. At least one, ideally two, of the four cached layers must be
full-attention layers -- this module enforces and logs that choice.

Run this once against the actual config.json on the GPU server before
Pass B (activation extraction) starts, and commit the resulting
layer_selection.json into the results directory / config so every run
downstream references the same indices.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerChoice:
    index: int
    depth_fraction: float
    attn_type: str  # "full_attention" | "linear_attention"


def classify_layers(hf_config: dict) -> list[str]:
    """Return a list of length num_layers, one of
    {"full_attention", "linear_attention"} per index, read from the HF
    config.json layer_types field (name may vary by release -- check the
    actual config.json on the server and adjust the key below if needed).
    """
    layer_types = hf_config.get("layer_types")
    if layer_types is None:
        raise KeyError(
            "hf_config has no 'layer_types' field. Open config.json on the "
            "server, find the field distinguishing Gated DeltaNet "
            "(linear-attention) from Gated Attention (full-attention) "
            "layers, and either rename it to 'layer_types' or update this "
            "function's key lookup to match. Do not guess the split from "
            "depth alone -- Qwen3.8-27B's 16 full-attention layers are "
            "interleaved, not just the last 16."
        )
    normalized = []
    for t in layer_types:
        t_low = str(t).lower()
        if "full" in t_low or t_low in ("attention", "full_attention"):
            normalized.append("full_attention")
        elif "linear" in t_low or "delta" in t_low:
            normalized.append("linear_attention")
        else:
            raise ValueError(f"Unrecognized layer_type value: {t!r}")
    return normalized


def select_cache_layers(
    layer_types: list[str],
    depth_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    min_full_attention: int = 2,
) -> list[LayerChoice]:
    """Pick one layer index per requested depth fraction, snapping each
    fraction to the nearest layer, but guaranteeing at least
    `min_full_attention` of the chosen layers are full-attention.

    Strategy: take the naive depth-fraction picks first; if fewer than
    `min_full_attention` of them are full-attention layers, swap the
    naive linear-attention picks (starting from the ones closest to a
    full-attention layer) for the nearest full-attention layer instead.
    """
    n = len(layer_types)
    full_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    if len(full_idx) < min_full_attention:
        raise ValueError(
            f"Only {len(full_idx)} full-attention layers exist; cannot "
            f"guarantee {min_full_attention} in the cache set."
        )

    naive = [min(n - 1, round(f * (n - 1))) for f in depth_fractions]

    def nearest_full(idx: int) -> int:
        return min(full_idx, key=lambda j: abs(j - idx))

    chosen = list(naive)
    n_full_in_choice = sum(1 for i in chosen if layer_types[i] == "full_attention")

    # Swap linear-attention picks (closest-to-a-full-layer first) until
    # the minimum is met.
    linear_positions = sorted(
        [p for p, i in enumerate(chosen) if layer_types[i] != "full_attention"],
        key=lambda p: min(abs(chosen[p] - j) for j in full_idx),
    )
    for p in linear_positions:
        if n_full_in_choice >= min_full_attention:
            break
        candidate = nearest_full(chosen[p])
        if candidate in chosen:
            # avoid duplicate picks; fall back to next-nearest full layer
            remaining = [j for j in full_idx if j not in chosen]
            if not remaining:
                continue
            candidate = min(remaining, key=lambda j: abs(j - chosen[p]))
        chosen[p] = candidate
        n_full_in_choice += 1

    if n_full_in_choice < min_full_attention:
        raise RuntimeError(
            "Could not satisfy min_full_attention after swapping -- "
            "inspect layer_types manually."
        )

    return [
        LayerChoice(index=idx, depth_fraction=frac, attn_type=layer_types[idx])
        for idx, frac in zip(chosen, depth_fractions)
    ]


def write_layer_selection(choices: list[LayerChoice], out_path: str) -> None:
    payload = {
        "chosen_layers": [
            {"index": c.index, "depth_fraction": c.depth_fraction, "attn_type": c.attn_type}
            for c in choices
        ],
        "note": (
            "Attention-dilution motivation (protocol §7.1) is a softmax-"
            "attention story; Gated DeltaNet linear-attention layers do "
            "not do the same normalized all-to-all competition. Whether "
            "G decays similarly at linear vs. full layers is an open "
            "empirical question -- report as a findings subsection, not "
            "just a threat-to-validity line."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
