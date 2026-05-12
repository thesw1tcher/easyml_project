from __future__ import annotations

from typing import Protocol, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from domain.project import Project


class ProjectRepository(Protocol):
    """
    Интерфейс для работы с проектами.
    """
    def save(self, project: Project, df: Optional[pd.DataFrame] = None) -> None:
        """Сохранить проект и, опционально, данные."""
        ...

    def load(self, project_id: str) -> Project:
        """Загрузить проект по ID."""
        ...

    def delete(self, project_id: str) -> None:
        """Удалить проект."""
        ...

    def list_projects(self) -> list[Project]:
        """Получить список всех проектов."""
        ...
