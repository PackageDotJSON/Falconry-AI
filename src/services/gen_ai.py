"""
@author: { MUHAMMAD HASSAN KIYANI, SOFTWARE DEVELOPER, FALCONRY SOLUTIONS }
@description: This file contains business logic related to Gen AI
"""

from functools import lru_cache
from io import BytesIO

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docling.exceptions import ConversionError
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool


@lru_cache(maxsize=1)
def _get_document_converter() -> DocumentConverter:
    return DocumentConverter()


def _convert_bytes_to_string(content: bytes, filename: str) -> str:
    if not content:
        raise ValueError("Uploaded file is empty")
    name = (filename or "document").strip() or "document"
    stream = DocumentStream(name=name, stream=BytesIO(content))
    try:
        result = _get_document_converter().convert(stream)
    except ConversionError as exc:
        raise ValueError(f"Document conversion failed: {exc!s}") from exc
    return result.document.export_to_markdown()


async def convert_document(file: UploadFile) -> str:
    content = await file.read()
    return await run_in_threadpool(
        _convert_bytes_to_string,
        content,
        file.filename or "document",
    )