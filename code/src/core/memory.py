import os
import json
from typing import List, Dict, Tuple, Optional
from src.core.observability import Logger

logger = Logger("memory")

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


from src.core.config import settings

class ConversationalMemory:
    def __init__(self, session_id: str = "default", call_llm_fn = None):
        self.session_id = session_id
        self.call_llm = call_llm_fn
        self.messages: List[Dict[str, str]] = [] # [{"role": "user"/"assistant", "content": "..."}]
        self.summary: str = ""
        
        # Initialize optional Redis backing
        self.redis_client = None
        self.redis_enabled = False
        
        redis_host = settings.REDIS_HOST
        redis_port = settings.REDIS_PORT
        if HAS_REDIS and redis_host:
            try:
                logger.info("Connecting to Redis", host=redis_host, port=redis_port)
                self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self.redis_enabled = True
                self.load_from_redis()
            except Exception as e:
                logger.warn("Redis connection failed. Conversations will be kept in-memory only.", error=str(e))
                self.redis_enabled = False

    def add_message(self, role: str, content: str):
        """Append user/assistant message to history and check if we should summarize."""
        self.messages.append({"role": role, "content": content})
        self.save_to_redis()
        
        # Trigger background summary if conversation gets long (e.g. > 10 messages)
        if len(self.messages) >= 10:
            self.summarize_history()

    def get_messages(self, limit: int = 6) -> List[Dict[str, str]]:
        """Return the recent message history within limits."""
        return self.messages[-limit:]

    def get_history_string(self) -> str:
        """Format the summary and recent messages as a clean text block."""
        history_parts = []
        if self.summary:
            history_parts.append(f"System Summary of past conversation: {self.summary}\n")
            
        for msg in self.messages[-6:]:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            history_parts.append(f"{role_label}: {msg['content']}")
            
        return "\n".join(history_parts)

    def summarize_history(self):
        """Condense old parts of conversation to keep prompt context sizes low."""
        if not self.call_llm or len(self.messages) < 6:
            return
            
        logger.info("Summarizing conversation history...")
        # Summarize messages before the last 4 messages
        messages_to_summarize = self.messages[:-4]
        history_to_summarize = ""
        for msg in messages_to_summarize:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_to_summarize += f"{role}: {msg['content']}\n"
            
        prompt = f"""Summarize the following conversation history concisely in 2-3 sentences.
Focus on key topics discussed and any entities mentioned.

Conversation:
{history_to_summarize}

Summary:"""
        try:
            provider, summary_text = self.call_llm(prompt, "You are a helpful assistant that summarizes conversations.")
            self.summary = summary_text.strip()
            # Prune messages: keep the summary and remove summarized messages
            self.messages = self.messages[-4:]
            self.save_to_redis()
            logger.info("Conversation summarized successfully", provider=provider)
        except Exception as e:
            logger.error("Failed to summarize conversation history", error=str(e))

    def save_to_redis(self):
        """Persist session state to Redis."""
        if not self.redis_enabled or not self.redis_client:
            return
            
        try:
            key = f"session:{self.session_id}:history"
            data = {
                "summary": self.summary,
                "messages": self.messages
            }
            self.redis_client.set(key, json.dumps(data))
        except Exception as e:
            logger.error("Failed to save history to Redis", error=str(e))

    def load_from_redis(self):
        """Retrieve session state from Redis."""
        if not self.redis_enabled or not self.redis_client:
            return
            
        try:
            key = f"session:{self.session_id}:history"
            raw_data = self.redis_client.get(key)
            if raw_data:
                data = json.loads(raw_data)
                self.summary = data.get("summary", "")
                self.messages = data.get("messages", [])
                logger.info("Loaded conversation session from Redis", session_id=self.session_id, count=len(self.messages))
        except Exception as e:
            logger.error("Failed to load history from Redis", error=str(e))

    def clear(self):
        """Reset conversation session."""
        self.messages = []
        self.summary = ""
        if self.redis_enabled and self.redis_client:
            try:
                key = f"session:{self.session_id}:history"
                self.redis_client.delete(key)
            except Exception as e:
                logger.error("Failed to delete session from Redis", error=str(e))
