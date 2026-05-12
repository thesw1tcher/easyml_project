import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Загрузка конфигурации из YAML файла
    """
    path = Path(config_path)
    if not path.exists():
        return {
            "app": {
                "title": "EasyML",
                "projects_root": "projects",
                "preview_limit": 200
            },
            "data_processing": {
                "fill_methods": ["Mean", "Median", "Constant"],
                "supported_formats": ["csv", "xlsx", "xls"]
            },
            "ml": {
                "task_types": ["classification", "regression"]
            }
        }
    
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

config = load_config()
