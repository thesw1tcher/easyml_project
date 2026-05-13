from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStatus(StrEnum):
    EMPTY = "empty"
    DATASET_LOADED = "dataset_loaded"
    READY_FOR_ML = "ready_for_ml"


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class Project:
    """
    Описывает все данные о проекте.
    """
    project_id: str
    name: str
    created_at: str
    updated_at: str
    status: ProjectStatus = ProjectStatus.EMPTY
    dataset_filename: Optional[str] = None
    dataset_snapshot_path: Optional[str] = None
    n_rows: int = 0
    n_cols: int = 0
    column_names: list[str] = field(default_factory=list)
    target_column: Optional[str] = None
    task_type: Optional[TaskType] = None
    feature_columns: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        """
        Валидация данных проекта.
        """
        if self.n_rows < 0:
            raise ValueError("n_rows cannot be negative")
        if self.n_cols < 0:
            raise ValueError("n_cols cannot be negative")
        if self.target_column and self.target_column in self.feature_columns:
            raise ValueError("target_column cannot be in feature_columns")

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
        data = asdict(self)
        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Project":
        status_val = data.get("status", "empty")
        if isinstance(status_val, str):
            try:
                status = ProjectStatus(status_val)
            except ValueError:
                status = ProjectStatus.EMPTY
        else:
            status = ProjectStatus.EMPTY

        task_type_val = data.get("task_type")
        task_type = None
        if task_type_val:
            try:
                task_type = TaskType(task_type_val)
            except ValueError:
                task_type = None

        return Project(
            project_id=data["project_id"],
            name=data["name"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            status=status,
            dataset_filename=data.get("dataset_filename"),
            dataset_snapshot_path=data.get("dataset_snapshot_path"),
            n_rows=int(data.get("n_rows", 0)),
            n_cols=int(data.get("n_cols", 0)),
            column_names=list(data.get("column_names", [])),
            target_column=data.get("target_column"),
            task_type=task_type,
            feature_columns=list(data.get("feature_columns", [])),
            notes=data.get("notes", ""),
        )
