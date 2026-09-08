"""Resume-aware per-run checkpointing and incremental result storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable


class CheckpointManager:
    def __init__(self, results_root: Path, pipeline: str) -> None:
        self.results_root = results_root
        self.pipeline = pipeline
        self.ckpt_dir = results_root / "raw" / "checkpoints"
        self.fail_dir = results_root / "raw" / "failed"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.fail_dir.mkdir(parents=True, exist_ok=True)

    def key(self, *parts: object) -> str:
        return "|".join(str(p) for p in parts)

    def _path(self, key: str) -> Path:
        safe = key.replace("|", "__").replace("/", "_")
        return self.ckpt_dir / f"{safe}.json"

    def is_complete(self, key: str) -> bool:
        return self._path(key).exists()

    def mark_complete(self, key: str, payload: dict) -> None:
        payload["_key"] = key
        payload["_pipeline"] = self.pipeline
        payload["_completed_at"] = datetime.now(timezone.utc).isoformat()
        with open(self._path(key), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str)

    def mark_failed(self, key: str, error: str) -> None:
        rec = {
            "key": key,
            "pipeline": self.pipeline,
            "error": str(error)[:2000],
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }
        path = self.fail_dir / f"{key.replace('|', '__').replace('/', '_')}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, default=str)

    def completed_keys(self) -> set:
        return {json.loads(p.read_text())["_key"] for p in self.ckpt_dir.glob("*.json")}

    def load_payloads(self) -> Iterable[dict]:
        for p in sorted(self.ckpt_dir.glob("*.json")):
            yield json.loads(p.read_text())

    def update_progress(self, progress: dict) -> None:
        path = self.results_root / "progress.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(progress, fh, indent=2, default=str)

    def write_progress_table(self, expected: Iterable[str], completed: set, failed: set) -> dict:
        exp = list(expected)
        progress = {
            "expected_runs": len(exp),
            "completed_runs": len(completed),
            "failed_runs": len(failed),
            "skipped_runs": len(exp) - len(completed) - len(failed),
            "last_completed_key": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.update_progress(progress)
        return progress

