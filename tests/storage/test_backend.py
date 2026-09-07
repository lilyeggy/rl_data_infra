from __future__ import annotations

import tempfile
import unittest

from src.errors import ContractValidationError
from src.storage import LocalFilesystemObjectBackend, PartitionKey


class StorageBackendTest(unittest.TestCase):
    def test_immutable_raw_object_is_idempotent_and_conflicts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = LocalFilesystemObjectBackend(temporary, durable=True)

            first = backend.put_raw("run-a/episode-a/raw-events.jsonl", b"one")
            second = backend.put_raw("run-a/episode-a/raw-events.jsonl", b"one")

            self.assertEqual(first, second)
            self.assertEqual(backend.get_raw("run-a/episode-a/raw-events.jsonl"), b"one")
            with self.assertRaises(ContractValidationError):
                backend.put_raw("run-a/episode-a/raw-events.jsonl", b"different")

    def test_manifest_is_replaceable_but_raw_object_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = LocalFilesystemObjectBackend(temporary, durable=True)
            key = "run-a/episode-a/partition-manifest.json"

            first = backend.put_manifest(key, b"manifest-v1")
            second = backend.put_manifest(key, b"manifest-v2")

            self.assertNotEqual(first, second)
            self.assertEqual(backend.get_manifest(key), b"manifest-v2")

    def test_object_keys_and_partition_identity_cannot_escape_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            backend = LocalFilesystemObjectBackend(temporary, durable=False)
            with self.assertRaises(ContractValidationError):
                backend.put_raw("../outside", b"bad")
            self.assertEqual(PartitionKey("run-a", "episode-a").value, "run-a/episode-a")


if __name__ == "__main__":
    unittest.main()
