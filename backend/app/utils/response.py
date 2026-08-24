from typing import Any

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse


def error_response(
    status_code: int,
    message: str,
    data: Any = None
) -> JSONResponse:

    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": message,
            "data": jsonable_encoder(data)
        }
    )
