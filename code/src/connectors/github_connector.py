import requests
from typing import List, Dict, Any
from src.connectors.connector_base import BaseConnector
from src.core.observability import Logger

logger = Logger("github_connector")

class GitHubConnector(BaseConnector):
    def __init__(self):
        self.token = None
        self.repo = None

    def authenticate(self, credentials: Dict[str, Any]) -> bool:
        self.token = credentials.get("token")
        self.repo = credentials.get("repo") # format: "owner/repo"
        return self.token is not None and self.repo is not None

    def sync(self) -> List[Dict[str, Any]]:
        if not self.token or not self.repo:
            raise PermissionError("GitHub connection unauthenticated.")
        logger.info("Starting GitHub repository sync...", repo=self.repo)
        return []

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search repository code/issues via GitHub search API endpoints."""
        if not self.token or not self.repo:
            logger.warn("GitHub connector unauthenticated. Simulating search results.")
            return [
                {
                    "source": f"github:{self.repo or 'mock/repo'}",
                    "text": f"Simulated GitHub code matching '{query}' in master branch.",
                    "page": 1,
                    "chapter": "Code base",
                    "section": "Main",
                    "id": "gh_1"
                }
            ]
            
        url = f"https://api.github.com/search/code?q={query}+repo:{self.repo}"
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json"
        }
        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                items = response.json().get("items", [])
                results = []
                for item in items[:limit]:
                    results.append({
                        "source": f"github:{item.get('repository', {}).get('full_name')}/{item.get('path')}",
                        "text": f"GitHub Search Match: {item.get('html_url')}",
                        "page": 1,
                        "chapter": "Code File",
                        "section": item.get("path"),
                        "id": f"gh_{item.get('sha', 'unknown')[:6]}"
                    })
                return results
            else:
                logger.warn("GitHub API returned non-200. Falling back to simulated results.")
        except Exception as e:
            logger.error("GitHub search request failed, falling back to simulated results", error=str(e))

        # Fallback simulated response
        return [
            {
                "source": f"github:{self.repo}",
                "text": f"Simulated GitHub code matching '{query}' in master branch.",
                "page": 1,
                "chapter": "Code base",
                "section": "Main",
                "id": "gh_1"
            }
        ]
