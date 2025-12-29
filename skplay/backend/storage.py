"""Azure Blob Storage integration for model and dataset storage."""

import io
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

import joblib

from skplay.config import settings

# Azure SDK is optional
try:
    from azure.storage.blob import (
        BlobClient,
        BlobSasPermissions,
        BlobServiceClient,
        ContainerClient,
        generate_blob_sas,
    )

    AZURE_STORAGE_AVAILABLE = True
except ImportError:
    AZURE_STORAGE_AVAILABLE = False


@dataclass
class StoredBlob:
    """Metadata for a stored blob."""

    blob_path: str
    container: str
    size_bytes: int
    content_type: str
    created_at: datetime
    url: str | None = None
    sas_url: str | None = None


class AzureBlobStorage:
    """Azure Blob Storage client for sklearn-playground."""

    def __init__(self):
        if not AZURE_STORAGE_AVAILABLE:
            raise RuntimeError(
                "azure-storage-blob package not installed. "
                "Install with: pip install azure-storage-blob"
            )

        conn_string = settings.azure_storage_connection_string.get_secret_value()
        if conn_string:
            self._service_client = BlobServiceClient.from_connection_string(conn_string)
        else:
            account_url = f"https://{settings.azure_storage_account}.blob.core.windows.net"
            self._service_client = BlobServiceClient(
                account_url=account_url,
                credential=settings.azure_storage_key.get_secret_value(),
            )

        self._account_name = settings.azure_storage_account
        self._account_key = settings.azure_storage_key.get_secret_value()

    def _get_container_client(self, container_name: str) -> "ContainerClient":
        """Get or create a container client."""
        container_client = self._service_client.get_container_client(container_name)

        # Create container if it doesn't exist
        if not container_client.exists():
            container_client.create_container()

        return container_client

    def upload_bytes(
        self,
        data: bytes | BinaryIO,
        container: str,
        blob_name: str | None = None,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredBlob:
        """Upload bytes to blob storage.

        Args:
            data: Bytes or file-like object to upload
            container: Container name
            blob_name: Blob name (auto-generated if not provided)
            content_type: MIME type
            metadata: Optional metadata

        Returns:
            StoredBlob with upload details
        """
        if blob_name is None:
            blob_name = f"{uuid.uuid4()}.bin"

        container_client = self._get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)

        # Handle BinaryIO
        if hasattr(data, "read"):
            data = data.read()

        blob_client.upload_blob(
            data,
            overwrite=True,
            content_settings={"content_type": content_type},
            metadata=metadata,
        )

        return StoredBlob(
            blob_path=blob_name,
            container=container,
            size_bytes=len(data),
            content_type=content_type,
            created_at=datetime.utcnow(),
            url=blob_client.url,
        )

    def download_bytes(self, container: str, blob_name: str) -> bytes:
        """Download blob as bytes."""
        container_client = self._get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        return blob_client.download_blob().readall()

    def download_to_stream(self, container: str, blob_name: str, stream: BinaryIO) -> int:
        """Download blob to a stream."""
        container_client = self._get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        download_stream = blob_client.download_blob()
        return download_stream.readinto(stream)

    def delete_blob(self, container: str, blob_name: str) -> bool:
        """Delete a blob."""
        try:
            container_client = self._get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.delete_blob()
            return True
        except Exception:
            return False

    def list_blobs(
        self,
        container: str,
        prefix: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List blobs in a container."""
        container_client = self._get_container_client(container)
        blobs = []

        for blob in container_client.list_blobs(name_starts_with=prefix):
            if len(blobs) >= limit:
                break
            blobs.append({
                "name": blob.name,
                "size": blob.size,
                "created": blob.creation_time,
                "modified": blob.last_modified,
                "content_type": blob.content_settings.content_type,
            })

        return blobs

    def generate_sas_url(
        self,
        container: str,
        blob_name: str,
        expiry_hours: int = 1,
        permissions: str = "r",
    ) -> str:
        """Generate a SAS URL for temporary access to a blob."""
        sas_token = generate_blob_sas(
            account_name=self._account_name,
            container_name=container,
            blob_name=blob_name,
            account_key=self._account_key,
            permission=BlobSasPermissions(read="r" in permissions, write="w" in permissions),
            expiry=datetime.utcnow() + timedelta(hours=expiry_hours),
        )

        return f"https://{self._account_name}.blob.core.windows.net/{container}/{blob_name}?{sas_token}"

    def blob_exists(self, container: str, blob_name: str) -> bool:
        """Check if a blob exists."""
        try:
            container_client = self._get_container_client(container)
            blob_client = container_client.get_blob_client(blob_name)
            return blob_client.exists()
        except Exception:
            return False


class LocalStorage:
    """Local file storage fallback for development."""

    def __init__(self, base_path: str = "./storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_container_path(self, container: str) -> Path:
        """Get path for a container."""
        path = self.base_path / container
        path.mkdir(parents=True, exist_ok=True)
        return path

    def upload_bytes(
        self,
        data: bytes | BinaryIO,
        container: str,
        blob_name: str | None = None,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> StoredBlob:
        """Upload bytes to local storage."""
        if blob_name is None:
            blob_name = f"{uuid.uuid4()}.bin"

        container_path = self._get_container_path(container)
        file_path = container_path / blob_name

        # Handle BinaryIO
        if hasattr(data, "read"):
            data = data.read()

        file_path.write_bytes(data)

        return StoredBlob(
            blob_path=blob_name,
            container=container,
            size_bytes=len(data),
            content_type=content_type,
            created_at=datetime.utcnow(),
            url=str(file_path),
        )

    def download_bytes(self, container: str, blob_name: str) -> bytes:
        """Download from local storage."""
        container_path = self._get_container_path(container)
        file_path = container_path / blob_name
        return file_path.read_bytes()

    def delete_blob(self, container: str, blob_name: str) -> bool:
        """Delete from local storage."""
        try:
            container_path = self._get_container_path(container)
            file_path = container_path / blob_name
            file_path.unlink()
            return True
        except Exception:
            return False

    def list_blobs(
        self,
        container: str,
        prefix: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List files in local storage."""
        container_path = self._get_container_path(container)
        blobs = []

        for file_path in container_path.iterdir():
            if prefix and not file_path.name.startswith(prefix):
                continue
            if len(blobs) >= limit:
                break

            stat = file_path.stat()
            blobs.append({
                "name": file_path.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
            })

        return blobs

    def blob_exists(self, container: str, blob_name: str) -> bool:
        """Check if file exists."""
        container_path = self._get_container_path(container)
        return (container_path / blob_name).exists()


def get_storage() -> AzureBlobStorage | LocalStorage:
    """Get the appropriate storage backend based on configuration."""
    if settings.is_azure_storage_configured:
        return AzureBlobStorage()
    return LocalStorage()


class ModelStorage:
    """High-level interface for storing and retrieving sklearn models."""

    def __init__(self, storage: AzureBlobStorage | LocalStorage | None = None):
        self.storage = storage or get_storage()
        self.container = settings.azure_storage_container_models

    def save_model(
        self,
        model: Any,
        model_name: str,
        user_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> StoredBlob:
        """Save a trained model to storage.

        Args:
            model: The trained sklearn model/pipeline
            model_name: Name for the model
            user_id: Optional user ID for organizing
            metadata: Optional metadata

        Returns:
            StoredBlob with storage details
        """
        # Serialize model
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)

        # Generate blob name
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        if user_id:
            blob_name = f"{user_id}/{model_name}_{timestamp}.joblib"
        else:
            blob_name = f"anonymous/{model_name}_{timestamp}.joblib"

        return self.storage.upload_bytes(
            data=buffer.read(),
            container=self.container,
            blob_name=blob_name,
            content_type="application/x-joblib",
            metadata=metadata,
        )

    def load_model(self, blob_path: str) -> Any:
        """Load a model from storage.

        Args:
            blob_path: Path to the blob

        Returns:
            The loaded model
        """
        data = self.storage.download_bytes(self.container, blob_path)
        buffer = io.BytesIO(data)
        return joblib.load(buffer)

    def list_models(
        self,
        user_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List stored models."""
        prefix = f"{user_id}/" if user_id else None
        return self.storage.list_blobs(self.container, prefix=prefix, limit=limit)

    def delete_model(self, blob_path: str) -> bool:
        """Delete a model from storage."""
        return self.storage.delete_blob(self.container, blob_path)
