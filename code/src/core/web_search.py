import os
import requests
from typing import List, Dict, Any
from src.core.observability import Logger

logger = Logger("web_search")

class WebSearchClient:
    def __init__(self):
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.serper_key = os.getenv("SERPER_API_KEY")
        
        if self.tavily_key:
            logger.info("Web Search Initialized with Tavily")
        elif self.serper_key:
            logger.info("Web Search Initialized with Serper")
        else:
            logger.warn("No Web Search API keys found (TAVILY_API_KEY or SERPER_API_KEY). Web fallback will run in simulation mode.")

    def search(self, query: str, num_results: int = 4) -> List[Dict[str, Any]]:
        """Query Tavily or Serper API and return search hits structured as RAG chunks."""
        if self.tavily_key and self.tavily_key != "your_tavily_api_key_here":
            return self._query_tavily(query, num_results)
        elif self.serper_key and self.serper_key != "your_serper_api_key_here":
            return self._query_serper(query, num_results)
        else:
            return self._query_mock(query)

    def _query_tavily(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.tavily_key,
            "query": query,
            "search_depth": "advanced",
            "max_results": num_results
        }
        try:
            logger.info("Triggering Tavily Web Search", query=query)
            response = requests.post(url, json=payload, timeout=8)
            if response.status_code == 200:
                data = response.json()
                results = []
                for idx, item in enumerate(data.get("results", [])):
                    results.append({
                        "id": f"web_{idx}",
                        "doc_id": -1,
                        "source": f"Web: {item.get('title', 'Tavily Search')}",
                        "text": item.get("content", ""),
                        "url": item.get("url", ""),
                        "page": 1
                    })
                return results
            else:
                logger.error("Tavily search API failed", status_code=response.status_code, response=response.text)
                return []
        except Exception as e:
            logger.error("Exception during Tavily search", error=str(e))
            return []

    def _query_serper(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": num_results
        }
        try:
            logger.info("Triggering Serper Web Search", query=query)
            response = requests.post(url, headers=headers, json=payload, timeout=8)
            if response.status_code == 200:
                data = response.json()
                results = []
                for idx, item in enumerate(data.get("organic", [])):
                    results.append({
                        "id": f"web_{idx}",
                        "doc_id": -1,
                        "source": f"Web: {item.get('title', 'Google Search')}",
                        "text": item.get("snippet", ""),
                        "url": item.get("link", ""),
                        "page": 1
                    })
                return results
            else:
                logger.error("Serper search API failed", status_code=response.status_code, response=response.text)
                return []
        except Exception as e:
            logger.error("Exception during Serper search", error=str(e))
            return []

    def _query_mock(self, query: str) -> List[Dict[str, Any]]:
        logger.info("Mock Web Search Fallback triggered", query=query)
        return [
            {
                "id": "web_mock_1",
                "doc_id": -1,
                "source": "Web Fallback (Simulation Mode)",
                "text": f"This is simulated web information for query '{query}' showing how the pipeline merges external sources when local document confidence is low.",
                "url": "https://simulated-web-fallback.ai",
                "page": 1
            }
        ]
