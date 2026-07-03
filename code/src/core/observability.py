import time
import logging
import json
from typing import Dict, Any

# Simple standard logging with structured formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

class Logger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        
    def info(self, event: str, **kwargs):
        payload = {"event": event, **kwargs}
        self.logger.info(json.dumps(payload))
        
    def warn(self, event: str, **kwargs):
        payload = {"event": event, **kwargs}
        self.logger.warning(json.dumps(payload))
        
    def error(self, event: str, **kwargs):
        payload = {"event": event, **kwargs}
        self.logger.error(json.dumps(payload))

class TelemetryTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.start_time = 0.0
        self.latencies: Dict[str, float] = {}
        self.token_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        self.providers_used = []
        
    def start_span(self, name: str):
        self.latencies[name] = time.time()
        
    def end_span(self, name: str):
        if name in self.latencies:
            self.latencies[name] = time.time() - self.latencies[name]
            
    def record_tokens(self, prompt: int, completion: int):
        self.token_usage["prompt_tokens"] += prompt
        self.token_usage["completion_tokens"] += completion
        self.token_usage["total_tokens"] += (prompt + completion)

    def record_provider(self, provider: str):
        if provider not in self.providers_used:
            self.providers_used.append(provider)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "latencies_sec": {k: round(v, 3) for k, v in self.latencies.items()},
            "token_usage": self.token_usage,
            "providers": self.providers_used
        }

# Global telemetry tracker instance
telemetry = TelemetryTracker()
