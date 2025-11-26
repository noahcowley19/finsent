

import time
from typing import Any, Optional


class Cache:
    TTL_PRICE = 900  # 15 minutes
    TTL_FUNDAMENTAL = 3600  # 1 hour
    TTL_METADATA = 86400  # 24 hours
    TTL_DEFAULT = 900  # 15 minutes

    def __init__(self):
        self._cache = {}

    def _get_key(self, ticker: str, data_type: str) -> str:
        
        return f"{ticker.upper()}:{data_type}"

    def set(self, ticker: str, data_type: str, value: Any, ttl: Optional[int] = None) -> None:
      
        if ttl is None:
            ttl = self.TTL_DEFAULT

        key = self._get_key(ticker, data_type)
        self._cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl
        }

    def get(self, ticker: str, data_type: str) -> Optional[Any]:
       
        key = self._get_key(ticker, data_type)

        if key not in self._cache:
            return None

        entry = self._cache[key]

        if not self._is_entry_valid(entry):
            # Clean up expired entry
            del self._cache[key]
            return None

        return entry['value']

    def _is_entry_valid(self, entry: dict) -> bool:
        
        age = time.time() - entry['timestamp']
        return age < entry['ttl']

    def is_valid(self, ticker: str, data_type: str) -> bool:
        
        key = self._get_key(ticker, data_type)

        if key not in self._cache:
            return False

        return self._is_entry_valid(self._cache[key])

    def get_age(self, ticker: str, data_type: str) -> Optional[float]:
       
        key = self._get_key(ticker, data_type)

        if key not in self._cache:
            return None

        return time.time() - self._cache[key]['timestamp']

    def clear(self, ticker: Optional[str] = None) -> None:
       
        if ticker is None:
            self._cache = {}
        else:
            prefix = f"{ticker.upper()}:"
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for key in keys_to_delete:
                del self._cache[key]

    def clear_expired(self) -> int:
      
        expired_keys = [
            key for key, entry in self._cache.items()
            if not self._is_entry_valid(entry)
        ]

        for key in expired_keys:
            del self._cache[key]

        return len(expired_keys)

    def get_stats(self) -> dict:
       
        total = len(self._cache)
        valid = sum(1 for entry in self._cache.values() if self._is_entry_valid(entry))
        expired = total - valid

        return {
            'total_entries': total,
            'valid_entries': valid,
            'expired_entries': expired
        }

financial_cache = Cache()
