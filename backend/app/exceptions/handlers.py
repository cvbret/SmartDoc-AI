import logging

from fastapi import Request

from app.exceptions.base import AppException
from app.utils.response import error_response


logger = logging.getLogger(__name__)


async def app_exception_handler(
    request: Request,
    exc: AppException
):
    return error_response(
        exc.status_code,
        exc.message,
        exc.data
    )


async def unhandled_exception_handler(
    request: Request,
    exc: Exception
):
    logger.exception(
        "未处理的应用异常"
    )

    return error_response(
        500,
        "服务器内部错误"
    )
