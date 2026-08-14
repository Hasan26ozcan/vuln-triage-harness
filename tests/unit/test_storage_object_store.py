"""Unit tests for app/storage/object_store.py — MinIO/S3 client wrapper.

Covers every function:

* ``get_client()`` — env-var resolution and boto3.client call args.
* ``ensure_bucket()`` — create when absent, skip when present.
* ``put_json()`` — put_object call args and S3 URI return value.
* ``get_json()`` — get_object, body read, JSON parsing.

The S3 client is always injected (or mocked) so no real MinIO or network
is required.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

boto3 = pytest.importorskip("boto3")

from app.storage.object_store import (  # noqa: E402
    DEFAULT_BUCKET,
    ensure_bucket,
    get_client,
    get_json,
    put_json,
)

# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------


class TestGetClient:
    def test_default_endpoint(self):
        """get_client builds an S3 client with defaults from env."""
        mock_client = MagicMock()
        env_no_minio = {k: v for k, v in __import__("os").environ.items()
                        if not k.startswith("MINIO_")}
        with (
            patch.dict("os.environ", env_no_minio, clear=True),
            patch("app.storage.object_store.boto3.client", return_value=mock_client) as mock_boto,
        ):
            result = get_client()
            assert result is mock_client
            call_kwargs = mock_boto.call_args
            assert call_kwargs[0][0] == "s3"
            assert call_kwargs[1]["endpoint_url"] == "http://localhost:9000"
            assert call_kwargs[1]["aws_access_key_id"] == "vuln_triage"
            assert call_kwargs[1]["aws_secret_access_key"] == "vuln_triage_secret"
            assert call_kwargs[1]["region_name"] == "us-east-1"

    def test_env_overrides(self):
        """MINIO_ENDPOINT and credentials from env are used."""
        mock_client = MagicMock()
        with (
            patch.dict("os.environ", {
                "MINIO_ENDPOINT": "http://custom:9000",
                "MINIO_ACCESS_KEY": "mykey",
                "MINIO_SECRET_KEY": "mysecret",
            }),
            patch("app.storage.object_store.boto3.client", return_value=mock_client) as mock_boto,
        ):
            get_client()
            call_kwargs = mock_boto.call_args
            assert call_kwargs[1]["endpoint_url"] == "http://custom:9000"
            assert call_kwargs[1]["aws_access_key_id"] == "mykey"
            assert call_kwargs[1]["aws_secret_access_key"] == "mysecret"

    def test_config_is_s3v4(self):
        """The botocore Config uses s3v4 signature."""
        mock_client = MagicMock()
        env_no_minio = {k: v for k, v in __import__("os").environ.items()
                        if not k.startswith("MINIO_")}
        with (
            patch.dict("os.environ", env_no_minio, clear=True),
            patch("app.storage.object_store.boto3.client", return_value=mock_client) as mock_boto,
        ):
            get_client()
            config_arg = mock_boto.call_args[1]["config"]
            assert config_arg.signature_version == "s3v4"


# ---------------------------------------------------------------------------
# ensure_bucket
# ---------------------------------------------------------------------------


class TestEnsureBucket:
    def test_creates_bucket_when_absent(self):
        """When the bucket doesn't exist, create_bucket is called."""
        mock_client = MagicMock()
        mock_client.list_buckets.return_value = {"Buckets": []}
        ensure_bucket(client=mock_client, bucket="new-bucket")
        mock_client.create_bucket.assert_called_once_with(Bucket="new-bucket")

    def test_skips_when_bucket_exists(self):
        """When the bucket already exists, create_bucket is NOT called."""
        mock_client = MagicMock()
        mock_client.list_buckets.return_value = {
            "Buckets": [{"Name": "existing-bucket"}, {"Name": "other"}]
        }
        ensure_bucket(client=mock_client, bucket="existing-bucket")
        mock_client.create_bucket.assert_not_called()

    def test_uses_default_bucket(self):
        """When bucket arg is omitted, DEFAULT_BUCKET is used."""
        mock_client = MagicMock()
        mock_client.list_buckets.return_value = {"Buckets": []}
        ensure_bucket(client=mock_client)
        mock_client.create_bucket.assert_called_once_with(Bucket=DEFAULT_BUCKET)

    def test_no_buckets_key(self):
        """If list_buckets returns no 'Buckets' key, a default empty list
        is used (the .get('Buckets', []) fallback)."""
        mock_client = MagicMock()
        mock_client.list_buckets.return_value = {}
        ensure_bucket(client=mock_client, bucket="my-bucket")
        mock_client.create_bucket.assert_called_once_with(Bucket="my-bucket")


# ---------------------------------------------------------------------------
# put_json
# ---------------------------------------------------------------------------


class TestPutJson:
    def test_put_json_returns_s3_uri(self):
        """put_json stores the payload and returns the s3:// URI."""
        mock_client = MagicMock()
        payload = {"key": "value", "nested": [1, 2, 3]}
        uri = put_json("my-key", payload, client=mock_client, bucket="my-bucket")

        assert uri == "s3://my-bucket/my-key"
        mock_client.put_object.assert_called_once()
        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "my-bucket"
        assert call_kwargs["Key"] == "my-key"
        assert call_kwargs["ContentType"] == "application/json"
        assert json.loads(call_kwargs["Body"].decode("utf-8")) == payload

    def test_put_json_uses_default_bucket(self):
        """When bucket is omitted, DEFAULT_BUCKET is used."""
        mock_client = MagicMock()
        put_json("some-key", {}, client=mock_client)
        mock_client.put_object.assert_called_once()
        assert mock_client.put_object.call_args[1]["Bucket"] == DEFAULT_BUCKET


# ---------------------------------------------------------------------------
# get_json
# ---------------------------------------------------------------------------


class TestGetJson:
    def test_get_json_parses_response(self):
        """get_json reads the object body and parses JSON."""
        mock_client = MagicMock()
        payload = {"cwe_id": "CWE-89", "severity": "high"}
        mock_client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=json.dumps(payload).encode("utf-8")))
        }

        result = get_json("my-key", client=mock_client, bucket="my-bucket")
        assert result == payload
        mock_client.get_object.assert_called_once_with(Bucket="my-bucket", Key="my-key")

    def test_get_json_uses_default_bucket(self):
        """When bucket is omitted, DEFAULT_BUCKET is used."""
        mock_client = MagicMock()
        mock_client.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=b'{}'))
        }
        get_json("some-key", client=mock_client)
        assert mock_client.get_object.call_args[1]["Bucket"] == DEFAULT_BUCKET
