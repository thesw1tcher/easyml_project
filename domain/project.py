from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Project:
    """
    Описывает все данные о проекте.
    """
    project_id: str
    name: str
    created_at: str
    updated_at: str
    status: str = "empty"  # empty | dataset_loaded | ready_for_eda | ready_for_ml
    dataset_filename: Optional[str] = None
    dataset_snapshot_path: Optional[str] = None
    n_rows: int = 0
    n_cols: int = 0
    column_names: list[str] = field(default_factory=list)
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # classification | regression
    feature_columns: list[str] = field(default_factory=list)
    notes: str = ""

    @staticmethod
    def create(name: str) -> "Project":
        now = utc_now_iso()
        return Project(
            project_id=str(uuid.uuid4()),
            name=name.strip(),
            created_at=now,
            updated_at=now,
        )

    def touch(self) -> None:
        """
        Обновляет timestamp последнего изменения проекта на текущее время.
        """
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Project":
        return Project(
            project_id=data["project_id"],
            name=data["name"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=data.get("status", "empty"),
            dataset_filename=data.get("dataset_filename"),
            dataset_snapshot_path=data.get("dataset_snapshot_path"),
            n_rows=int(data.get("n_rows", 0)),
            n_cols=int(data.get("n_cols", 0)),
            column_names=list(data.get("column_names", [])),
            target_column=data.get("target_column"),
            task_type=data.get("task_type"),
            feature_columns=list(data.get("feature_columns", [])),
            notes=data.get("notes", ""),
        )
