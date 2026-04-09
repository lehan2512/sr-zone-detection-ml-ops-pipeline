import os
import logging
from pathlib import Path
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)

def get_blob_service_client():
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        logger.error("AZURE_STORAGE_CONNECTION_STRING not set.")
        return None
    return BlobServiceClient.from_connection_string(connection_string)

def upload_to_blob(file_path: Path, container_name: str, blob_name: str = None):
    """
    Uploads a local file to an Azure Blob Storage container.
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    if not blob_name:
        blob_name = file_path.name

    try:
        blob_service_client = get_blob_service_client()
        if not blob_service_client:
            return False
            
        container_client = blob_service_client.get_container_client(container_name)
        
        # Create container if it doesn't exist
        if not container_client.exists():
            logger.info(f"Creating container: {container_name}")
            container_client.create_container()

        blob_client = container_client.get_blob_client(blob_name)
        
        logger.info(f"Uploading {file_path} to blob {blob_name} in container {container_name}...")
        
        with open(file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)
        
        logger.info("Upload successful.")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to Azure Blob Storage: {str(e)}")
        return False

def download_blob(container_name: str, blob_name: str, download_path: Path):
    """
    Downloads a blob from Azure Blob Storage to a local file.
    """
    try:
        blob_service_client = get_blob_service_client()
        if not blob_service_client:
            return False

        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        
        if not blob_client.exists():
            logger.warning(f"Blob {blob_name} does not exist in container {container_name}.")
            return False

        logger.info(f"Downloading blob {blob_name} from container {container_name} to {download_path}...")
        
        download_path.parent.mkdir(parents=True, exist_ok=True)
        with open(download_path, "wb") as file:
            download_stream = blob_client.download_blob()
            file.write(download_stream.readall())
        
        logger.info("Download successful.")
        return True
    except Exception as e:
        logger.error(f"Failed to download from Azure Blob Storage: {str(e)}")
        return False
