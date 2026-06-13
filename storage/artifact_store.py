"""
storage/artifact_store.py
Pluggable artifact store for large TrajectoryRecords.
Supports S3, GCS, and local filesystem backends.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ArtifactBackend(ABC):
    """Abstract backend for artifact storage."""

    @abstractmethod
    async def upload(self, key: str, data: bytes) -> str:
        """Upload data and return the artifact URL."""
        raise NotImplementedError

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """Download data by key."""
        raise NotImplementedError


class LocalBackend(ArtifactBackend):
    """Local filesystem backend (default fallback)."""

    def __init__(self, base_dir: str = "./artifacts"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    async def upload(self, key: str, data: bytes) -> str:
        path = os.path.join(self.base_dir, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return f"file://{os.path.abspath(path)}"

    async def download(self, key: str) -> bytes:
        path = os.path.join(self.base_dir, key)
        with open(path, "rb") as f:
            return f.read()


class S3Backend(ArtifactBackend):
    """AWS S3 backend using boto3."""

    def __init__(
        self,
        bucket: str = "agent-eval-artifacts",
        region: str = "us-east-1",
        prefix: str = "trajectories/",
    ):
        self.bucket = bucket
        self.region = region
        self.prefix = prefix

    async def upload(self, key: str, data: bytes) -> str:
        import boto3

        s3 = boto3.client("s3", region_name=self.region)
        full_key = f"{self.prefix}{key}"
        s3.put_object(Bucket=self.bucket, Key=full_key, Body=data)
        return f"s3://{self.bucket}/{full_key}"

    async def download(self, key: str) -> bytes:
        import boto3

        s3 = boto3.client("s3", region_name=self.region)
        full_key = f"{self.prefix}{key}"
        response = s3.get_object(Bucket=self.bucket, Key=full_key)
        return response["Body"].read()


class GCSBackend(ArtifactBackend):
    """Google Cloud Storage backend."""

    def __init__(
        self,
        bucket: str = "agent-eval-artifacts",
        prefix: str = "trajectories/",
    ):
        self.bucket = bucket
        self.prefix = prefix

    async def upload(self, key: str, data: bytes) -> str:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(self.bucket)
        full_key = f"{self.prefix}{key}"
        blob = bucket.blob(full_key)
        blob.upload_from_string(data)
        return f"gs://{self.bucket}/{full_key}"

    async def download(self, key: str) -> bytes:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(self.bucket)
        full_key = f"{self.prefix}{key}"
        blob = bucket.blob(full_key)
        return blob.download_as_bytes()


class ArtifactStore:
    """
    High-level artifact store with pluggable backend.

    Used to offload large TrajectoryRecord JSON from PostgreSQL
    to a scalable object store, keeping only the artifact_url in the DB.
    """

    def __init__(self, backend: Optional[ArtifactBackend] = None):
        self.backend = backend or LocalBackend()

    async def upload_trajectory(
        self,
        report_id: str,
        trajectory_data: Dict[str, Any],
    ) -> str:
        """
        Upload trajectory data and return the artifact URL.

        Args:
            report_id: Unique identifier for the report.
            trajectory_data: Full trajectory data as a dict.

        Returns:
            Artifact URL (s3://, gs://, or file://).
        """
        key = f"{report_id}.json"
        data = json.dumps(trajectory_data, indent=2, default=str).encode("utf-8")
        return await self.backend.upload(key, data)

    async def download_trajectory(self, report_id: str) -> Dict[str, Any]:
        """Download and parse trajectory data by report_id."""
        key = f"{report_id}.json"
        data = await self.backend.download(key)
        return json.loads(data.decode("utf-8"))
