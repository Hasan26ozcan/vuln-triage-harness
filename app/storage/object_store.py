"""MinIO/S3 client wrapper.

Full VulnSample JSON (including the actual code, which can be large) goes
to object storage; Postgres only keeps queryable metadata plus the object
key (see VulnSampleRow.object_store_key). This split is deliberate — it
keeps the DB small and fast to query while the bulky payloads live where
bulk storage belongs.
"""

from __future__ import annotations

import json
import os
from typing import Any

import boto3
from botocore.client import Config

DEFAULT_BUCKET = "vuln-triage"


def get_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.environ.get("MINIO_ACCESS_KEY", "vuln_triage"),
        aws_secret_access_key=os.environ.get("MINIO_SECRET_KEY", "vuln_triage_secret"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(client: Any | None = None, bucket: str = DEFAULT_BUCKET) -> None:
    client = client or get_client()
    existing = {b["Name"] for b in client.list_buckets().get("Buckets", [])}
    if bucket not in existing:
        client.create_bucket(Bucket=bucket)


def put_json(
    key: str,
    payload: dict[str, Any],
    client: Any | None = None,
    bucket: str = DEFAULT_BUCKET,
) -> str:
    client = client or get_client()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def get_json(key: str, client: Any | None = None, bucket: str = DEFAULT_BUCKET) -> dict[str, Any]:
    client = client or get_client()
    obj = client.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())
