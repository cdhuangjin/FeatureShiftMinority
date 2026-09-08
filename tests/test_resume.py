"""Checkpoint manager resume semantics."""

from pathlib import Path

from src.checkpoint_manager import CheckpointManager


def test_resume_skips_completed(tmp_path):
    cm = CheckpointManager(Path(tmp_path), "test")
    k = cm.key("d", 42, "xgboost", "c12_cell")
    assert not cm.is_complete(k)
    cm.mark_complete(k, {"perf": [1, 2]})
    assert cm.is_complete(k)
    assert k in cm.completed_keys()
    # A completed key is skipped when the run re-executes with resume.


def test_failed_recorded(tmp_path):
    cm = CheckpointManager(Path(tmp_path), "test")
    k = cm.key("d", 42, "rf", "c12_cell")
    cm.mark_failed(k, "boom")
    assert len(list(cm.fail_dir.glob("*.json"))) == 1
    assert not cm.is_complete(k)

