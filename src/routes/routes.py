"""
@author: { MUHAMMAD HASSAN KIYANI, SOFTWARE DEVELOPER, FALCONRY SOLUTIONS }
@description: This file contains API endpoint related to this application
"""

from fastapi import APIRouter, Depends, File, HTTPException, Security, UploadFile, status
from fastapi.security import APIKeyHeader

import configs.configs as config
from enums.enums import GenAIUrls
from services.gen_ai import convert_document

gen_ai_router = APIRouter(
    prefix=GenAIUrls.ROUTE_PREFIX.value,
    tags=GenAIUrls.TAGS.value,
)

api_key_header = APIKeyHeader(name=config.API_KEY_HEADER_NAME, auto_error=False)


def validate_api_key(api_key: str | None = Security(api_key_header)) -> None:
    if not config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key is not configured on server.",
        )
    if api_key != config.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


@gen_ai_router.post(GenAIUrls.CONVERT_DOCUMENT.value)
async def convert_document_api(
    _: None = Depends(validate_api_key),
    file: UploadFile = File(
        ...,
        description="Source document (e.g. PDF, DOCX, PPTX, HTML, images; formats supported by Docling).",
    ),
):
    try:
        text = await convert_document(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"text": text, "filename": file.filename}
