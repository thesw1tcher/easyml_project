class EasyMLError(Exception):
    """Base exception for EasyML."""
    pass


class ProjectNotFoundError(EasyMLError):
    """Raised when a project is not found in the repository."""
    pass


class DatasetLoadError(EasyMLError):
    """Raised when there is an error loading a dataset."""
    pass
