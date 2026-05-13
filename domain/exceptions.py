class EasyMLError(Exception):
    """Базовое исключение для EasyML."""
    pass


class ProjectNotFoundError(EasyMLError):
    """Исключение возникает, когда проект не найден в репозитории."""
    pass


class DatasetLoadError(EasyMLError):
    """Исключение возникает при загрузке датасета."""
    pass
