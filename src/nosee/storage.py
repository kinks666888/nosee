"""Small settings store plus a Keychain-backed owner face embedding."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .paths import APP_SUPPORT, KEYCHAIN_ACCOUNT, KEYCHAIN_SERVICE, ensure_app_support

SETTINGS_PATH = APP_SUPPORT / "settings.json"


class StorageError(RuntimeError):
    pass


class LocalStore:
    def __init__(self) -> None:
        ensure_app_support()

    def load_settings(self) -> dict[str, Any]:
        try:
            data = json.loads(SETTINGS_PATH.read_text("utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save_settings(self, settings: dict[str, Any]) -> None:
        ensure_app_support()
        temp = SETTINGS_PATH.with_suffix(".tmp")
        temp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), "utf-8")
        temp.chmod(0o600)
        temp.replace(SETTINGS_PATH)

    @staticmethod
    def _security():
        try:
            import Security
        except ImportError as exc:
            raise StorageError("Keychain bridge is unavailable. Install pyobjc-framework-Security.") from exc
        return Security

    def save_owner_embedding(
        self,
        values: list[float],
        threshold: float = 0.42,
        embedding_version: int = 2,
    ) -> None:
        if not values or not all(math.isfinite(float(v)) for v in values):
            raise StorageError("The face embedding is empty or invalid.")
        if not math.isfinite(float(threshold)) or not 0.0 < float(threshold) < 1.0:
            raise StorageError("The owner match threshold is invalid.")
        Security = self._security()
        from Foundation import NSData

        record = {
            "version": 3,
            "embedding_version": int(embedding_version),
            "embedding": [float(v) for v in values],
            "threshold": float(threshold),
        }
        payload = json.dumps(record, separators=(",", ":")).encode("utf-8")
        data = NSData.dataWithBytes_length_(payload, len(payload))
        query = {
            Security.kSecClass: Security.kSecClassGenericPassword,
            Security.kSecAttrService: KEYCHAIN_SERVICE,
            Security.kSecAttrAccount: KEYCHAIN_ACCOUNT,
        }
        # SecItemDelete takes only the query dictionary (unlike the add/copy
        # functions, whose result is an output parameter in the C API).
        Security.SecItemDelete(query)
        query.update(
            {
                Security.kSecValueData: data,
                Security.kSecAttrAccessible: Security.kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            }
        )
        result = Security.SecItemAdd(query, None)
        status = result[0] if isinstance(result, tuple) else result
        if status != Security.errSecSuccess:
            raise StorageError(f"Could not save owner data to Keychain (status {status}).")

    def load_owner_profile(self) -> tuple[list[float] | None, float, int]:
        Security = self._security()
        query = {
            Security.kSecClass: Security.kSecClassGenericPassword,
            Security.kSecAttrService: KEYCHAIN_SERVICE,
            Security.kSecAttrAccount: KEYCHAIN_ACCOUNT,
            Security.kSecReturnData: True,
            Security.kSecMatchLimit: Security.kSecMatchLimitOne,
        }
        result = Security.SecItemCopyMatching(query, None)
        if isinstance(result, tuple):
            status, value = result
        else:
            status, value = result, None
        if status == Security.errSecItemNotFound:
            return None, 0.42, 1
        if status != Security.errSecSuccess or value is None:
            raise StorageError(f"Could not read owner data from Keychain (status {status}).")
        try:
            raw = bytes(value)
            record = json.loads(raw.decode("utf-8"))
            if isinstance(record, dict):
                values = record.get("embedding")
                threshold = float(record.get("threshold", 0.42))
                embedding_version = int(record.get("embedding_version", 1))
            else:
                # Backward compatibility with embeddings saved by NoSee 0.1.0.
                values = record
                threshold = 0.42
                embedding_version = 1
            parsed = [float(v) for v in values]
            if not parsed or not all(math.isfinite(v) for v in parsed):
                raise ValueError("empty or non-finite embedding")
            if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
                raise ValueError("invalid threshold")
            return parsed, threshold, embedding_version
        except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StorageError("Owner profile in Keychain is not readable.") from exc

    def load_owner_embedding(self) -> list[float] | None:
        return self.load_owner_profile()[0]

    def clear_owner_embedding(self) -> None:
        Security = self._security()
        query = {
            Security.kSecClass: Security.kSecClassGenericPassword,
            Security.kSecAttrService: KEYCHAIN_SERVICE,
            Security.kSecAttrAccount: KEYCHAIN_ACCOUNT,
        }
        status = Security.SecItemDelete(query)
        if status not in (Security.errSecSuccess, Security.errSecItemNotFound):
            raise StorageError(f"Could not delete owner data from Keychain (status {status}).")

