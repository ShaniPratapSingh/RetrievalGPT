from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseConnector(ABC):
    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        """Authenticate with the target connector source."""
        pass

    @abstractmethod
    def sync(self) -> List[Dict[str, Any]]:
        """Incrementally sync and pull data/documents from the source."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search the target data source directly."""
        pass
