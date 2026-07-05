import os
import hashlib
import json
from typing import Optional, List, Dict, Any
from src.core.observability import Logger

logger = Logger("cache")

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


from src.core.config import settings

class RAGCache:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = os.path.abspath(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Redis setup
        self.redis_client = None
        self.redis_enabled = False
        
        redis_host = settings.REDIS_HOST
        redis_port = settings.REDIS_PORT
        if HAS_REDIS and redis_host:
            try:
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self.redis_enabled = True
                logger.info("Cache connected to Redis")
            except Exception as e:
                logger.warn("Cache fell back to disk-only because Redis connection failed", error=str(e))
                self.redis_enabled = False

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Retrieve embedding from Redis or disk cache."""
        h = self._get_hash(text)
        
        # 1. Try Redis
        if self.redis_enabled and self.redis_client:
            try:
                raw = self.redis_client.get(f"emb:{h}")
                if raw:
                    return json.loads(raw)
            except Exception as e:
                pass
                
        # 2. Try Disk
        disk_path = os.path.join(self.cache_dir, f"emb_{h}.json")
        if os.path.exists(disk_path):
            try:
                with open(disk_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
                
        return None

    def set_embedding(self, text: str, embedding: List[float]):
        """Save embedding to Redis and disk cache."""
        h = self._get_hash(text)
        
        # 1. Save to Redis
        if self.redis_enabled and self.redis_client:
            try:
                self.redis_client.set(f"emb:{h}", json.dumps(embedding), ex=86400 * 7) # Cache for 7 days
            except Exception:
                pass
                
        # 2. Save to Disk
        disk_path = os.path.join(self.cache_dir, f"emb_{h}.json")
        try:
            with open(disk_path, "w", encoding="utf-8") as f:
                json.dump(embedding, f)
        except Exception as e:
            logger.warn("Failed to write embedding cache to disk", error=str(e))

    def get_completion(self, prompt: str) -> Optional[str]:
        """Retrieve completion text from cache."""
        h = self._get_hash(prompt)
        
        if self.redis_enabled and self.redis_client:
            try:
                return self.redis_client.get(f"comp:{h}")
            except Exception:
                pass
                
        disk_path = os.path.join(self.cache_dir, f"comp_{h}.json")
        if os.path.exists(disk_path):
            try:
                with open(disk_path, "r", encoding="utf-8") as f:
                    return json.load(f).get("completion")
            except Exception:
                pass
                
        return None

    def set_completion(self, prompt: str, completion: str):
        """Save completion text to cache."""
        h = self._get_hash(prompt)
        
        if self.redis_enabled and self.redis_client:
            try:
                self.redis_client.set(f"comp:{h}", completion, ex=86400) # Cache for 24 hours
            except Exception:
                pass
                
        disk_path = os.path.join(self.cache_dir, f"comp_{h}.json")
        try:
            with open(disk_path, "w", encoding="utf-8") as f:
                json.dump({"completion": completion}, f)
        except Exception as e:
            logger.warn("Failed to write completion cache to disk", error=str(e))
