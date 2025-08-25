import json
from pathlib import Path
from typing import Dict, Any
import logging
import importlib.resources

logger = logging.getLogger(__name__)


class UserDataManager:
    def __init__(self, file_path: str = "./UserData.json"):
        self.file_path = Path(file_path)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """从JSON文件加载数据。如果文件不存在或无效，则返回一个空字典。"""
        if self.file_path.exists():
            try:
                with self.file_path.open('r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"无法读取用户数据文件 {self.file_path}。将使用空配置。错误：{e}")
                return {}
        return {}

    def _save(self):
        """将当前内存中的所有数据以美化格式保存到JSON文件。"""
        try:
            with self.file_path.open('w', encoding='utf-8') as f:
                json.dump(self._data, f, indent=4, ensure_ascii=False)  # type: ignore
        except IOError as e:
            logger.warning(f"无法写入用户数据文件 {self.file_path}。错误：{e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any):
        self._data[key] = value
        self._save()


userdata_file: str = str(importlib.resources.files('Genie_CPU') / 'UserData.json')
userdata_manager = UserDataManager(file_path=userdata_file)
