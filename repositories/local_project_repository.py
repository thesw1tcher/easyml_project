from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

from domain.project import Project


PROJECTS_ROOT = Path("projects")
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)


class LocalProjectRepository:
    """
    Реализует хранение в локальной файловой системе.
    """
    def __init__(self, root: Path = PROJECTS_ROOT) -> None:
        """Создаёт папку в projects/"""
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        """
        Путь до папки с проектом.
        """
        return self.root / project_id

    def meta_path(self, project_id: str) -> Path:
        """
        Глобальынй путь до папки с проектом.
        """
        return self.project_dir(project_id) / "project.json"

    def dataset_path(self, project_id: str, original_filename: str) -> Path:
        """
        Путь до таблицы с данными.
        """
        safe_name = Path(original_filename).name
        return self.project_dir(project_id) / safe_name

    def save(self, project: Project, df: Optional[pd.DataFrame] = None) -> None:
        """
        Кладёт в projects/project.project_id json-файл с данными проекта.
        """
        project.touch()
        project_folder = self.project_dir(project.project_id)
        project_folder.mkdir(parents=True, exist_ok=True)

        if df is not None and project.dataset_filename:
            snapshot_path = self.dataset_path(
                project.project_id, project.dataset_filename)
            df.to_csv(snapshot_path, index=False)
            project.dataset_snapshot_path = str(snapshot_path)
            project.n_rows = int(df.shape[0])
            project.n_cols = int(df.shape[1])
            project.column_names = df.columns.astype(str).tolist()
            project.status = "dataset_loaded"

        with self.meta_path(project.project_id).open("w", encoding="utf-8") as f:
            json.dump(project.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, project_id: str) -> Project:
        """
        Инициализирует проект из projects/project.project_id/project.json
        """
        with self.meta_path(project_id).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return Project.from_dict(data)

    def delete(self, project_id: str) -> None:
        shutil.rmtree(self.project_dir(project_id), ignore_errors=True)

    def list_projects(self) -> list[Project]:
        """
        Список проектов, расположенных в папке projects/
        """
        projects: list[Project] = []
        for meta_file in sorted(self.root.glob("*/project.json")):
            try:
                with meta_file.open("r", encoding="utf-8") as f:
                    projects.append(Project.from_dict(json.load(f)))
            except Exception:
                continue
        return sorted(projects, key=lambda p: p.updated_at, reverse=True)
