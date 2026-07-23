import json

import numpy as np
import pandas as pd
import pytest
import torch

from dataset_highlight import HighlightDataset
from model_highlight import (
    HighlightLoss,
    HierarchicalHighlightTransformer,
    build_model,
)


def test_dataset_aligns_segment_t_with_target_t_plus_1(tmp_path, monkeypatch):
    stat_cols = ["stat_rel_pos", "stat_seg_dur_log", "stat_cum_viewers"]
    stats = {
        "segment_emb_dim": 2,
        "stats_feat_dim": 3,
        "stats_feat_cols": stat_cols,
    }
    (tmp_path / "stats.json").write_text(json.dumps(stats))

    frame = pd.DataFrame(
        {
            "live_stream_id": [7, 7, 7],
            "timestamp": [10, 20, 30],
            "split": ["train", "train", "train"],
            "feature_128": [
                np.array([1.0, 10.0]),
                np.array([2.0, 20.0]),
                np.array([3.0, 30.0]),
            ],
            "hl_score": [0.1, 0.2, 0.3],
            "hl_binary": [0, 1, 0],
            "stat_rel_pos": [0.1, 0.2, 0.3],
            "stat_seg_dur_log": [1.0, 2.0, 3.0],
            # Row k is history through k-1.
            "stat_cum_viewers": [0.0, 10.0, 30.0],
        }
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _: frame)

    plain = HighlightDataset(str(tmp_path), "train", max_seq_len=200)[0]
    stats_only = HighlightDataset(
        str(tmp_path), "train", max_seq_len=200, input_mode="stats"
    )[0]
    emb_stats = HighlightDataset(
        str(tmp_path), "train", max_seq_len=200,
        input_mode="embedding", use_stats=True,
    )[0]

    torch.testing.assert_close(
        plain["embeddings"],
        torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
    )
    expected_stats = torch.tensor(
        [
            [0.1, 1.0, 10.0],
            [0.2, 2.0, 30.0],
        ]
    )
    torch.testing.assert_close(stats_only["embeddings"], expected_stats)
    torch.testing.assert_close(emb_stats["embeddings"], plain["embeddings"])
    torch.testing.assert_close(emb_stats["stats"], expected_stats)

    for sample in (plain, stats_only, emb_stats):
        torch.testing.assert_close(sample["hl_scores"], torch.tensor([0.2, 0.3]))
        torch.testing.assert_close(sample["hl_binary"], torch.tensor([1, 0]))
        assert sample["length"] == 2


def test_hierarchical_output_cannot_see_future_embeddings():
    torch.manual_seed(0)
    model = HierarchicalHighlightTransformer(
        seg_dim=2,
        d_model=8,
        n_heads=2,
        n_global=2,
        window=3,
        dropout=0.0,
        max_seq_len=8,
    ).eval()

    embeddings = torch.randn(1, 4, 2)
    changed = embeddings.clone()
    changed[:, 2:] += 100.0
    pad_mask = torch.zeros(1, 4, dtype=torch.bool)

    with torch.no_grad():
        original_scores = model(embeddings, pad_mask)
        changed_scores = model(changed, pad_mask)

    # Outputs at t=0 and t=1 predict segments 1 and 2. Neither may change when
    # only input positions 2 and later are perturbed.
    torch.testing.assert_close(
        original_scores[:, :2],
        changed_scores[:, :2],
        rtol=0.0,
        atol=1e-7,
    )


def test_all_models_and_input_variants_train_on_next_segment_targets():
    torch.manual_seed(1)
    embeddings = torch.randn(2, 4, 2)
    stats = torch.randn(2, 4, 3)
    targets = torch.rand(2, 4)
    pad_mask = torch.tensor(
        [[False, False, False, False], [False, False, True, True]]
    )
    targets = targets.masked_fill(pad_mask, -1.0)

    loss_fn = HighlightLoss()

    for variant in ("plain", "stats", "es"):
        seg_dim = 3 if variant == "stats" else 2
        model_input = stats if variant == "stats" else embeddings
        stat_input = stats if variant == "es" else None
        use_stats = variant == "es"

        model_kwargs = {
            "causal": dict(
                seg_dim=seg_dim, d_model=8, n_heads=2, n_layers=1,
                dropout=0.0, max_seq_len=8,
                use_stats=use_stats, stat_dim=3,
            ),
            "hierarchical": dict(
                seg_dim=seg_dim, d_model=8, n_heads=2, n_global=1, window=3,
                dropout=0.0, max_seq_len=8,
                use_stats=use_stats, stat_dim=3,
            ),
            "gru": dict(
                seg_dim=seg_dim, hidden=8, n_layers=1, dropout=0.0,
                use_stats=use_stats, stat_dim=3,
            ),
            "mlp": dict(
                seg_dim=seg_dim, hidden=8, dropout=0.0,
                use_stats=use_stats, stat_dim=3,
            ),
        }

        for name, kwargs in model_kwargs.items():
            model = build_model(name, **kwargs)
            predictions = model(model_input, pad_mask, stat_input)
            assert predictions.shape == targets.shape, (name, variant)
            loss = loss_fn(predictions, targets, pad_mask)
            assert torch.isfinite(loss), (name, variant)
            loss.backward()


def test_embedding_plus_stats_cannot_silently_drop_stats():
    model = build_model(
        "mlp", seg_dim=2, hidden=8, use_stats=True, stat_dim=3
    )
    with pytest.raises(ValueError, match="requires a stats tensor"):
        model(
            torch.randn(1, 2, 2),
            torch.zeros(1, 2, dtype=torch.bool),
            stats=None,
        )
