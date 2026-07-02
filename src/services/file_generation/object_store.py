"""
@author: { FALCONRY SOLUTIONS }
@description: Object store integration for file persistence.

Generated files are uploaded to S3 (or any S3-compatible store such as MinIO,
Cloudflare R2, or DigitalOcean Spaces) so they are available as static assets
after the API response has been sent.

Upload is entirely optional — if the required environment variables are absent
the upload is silently skipped and None is returned, so callers always get the
in-memory file bytes regardless of object store availability.

Required environment variables (when S3 upload is desired):
  AWS_S3_BUCKET           — target bucket name
  AWS_ACCESS_KEY_ID       — IAM / service-account access key
  AWS_SECRET_ACCESS_KEY   — IAM / service-account secret key

Optional:
  AWS_S3_REGION           — defaults to "us-east-1"
  AWS_S3_ENDPOINT_URL     — custom endpoint for MinIO / R2 / Spaces / etc.
  AWS_S3_PUBLIC_BASE_URL  — override the returned public URL prefix (e.g. CDN domain)
"""

import os
from typing import Optional


def _is_configured() -> bool:
    """Return True only when all required S3 env vars are present."""
    return all(
        os.environ.get(k)
        for k in ("AWS_S3_BUCKET", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
    )


def upload_file(
    file_bytes: bytes,
    object_key: str,
    content_type: str,
) -> Optional[str]:
    """
    Upload file_bytes to S3 under object_key and return a public URL.

    Returns None immediately if S3 is not configured so the rest of the
    application can proceed without object store support.

    Args:
        file_bytes:   Raw file content to upload.
        object_key:   S3 key / path inside the bucket (e.g. "reports/2026/report.pdf").
        content_type: MIME type string (e.g. "application/pdf").

    Returns:
        Public URL string if upload succeeded, None otherwise.
    """
    if not _is_configured():
        return None

    try:
        import boto3

        bucket = os.environ["AWS_S3_BUCKET"]
        region = os.environ.get("AWS_S3_REGION", "us-east-1")
        endpoint = os.environ.get("AWS_S3_ENDPOINT_URL")

        client = boto3.client(      
            "s3",
            region_name=region,
            aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            **({"endpoint_url": endpoint} if endpoint else {}),
        )

        client.put_object(
            Bucket=bucket,
            Key=object_key,
            Body=file_bytes,
            ContentType=content_type,
        )
   
        # Build the public URL — prefer a custom base URL (e.g. CDN) if provided
        base_url = os.environ.get("AWS_S3_PUBLIC_BASE_URL")
        if base_url:
            return f"{base_url.rstrip('/')}/{object_key}"

        if endpoint:
            return f"{endpoint.rstrip('/')}/{bucket}/{object_key}"

        return f"https://{bucket}.s3.{region}.amazonaws.com/{object_key}"

    except Exception:
        # Upload failure must never crash the API response — file bytes are
        # always returned to the caller regardless of object store state.
        return None

