from typing import Any


class AppException(Exception):

    def __init__(
        self,
        status_code: int,
        message: str,
        data: Any = None
    ):
        self.status_code = status_code
        self.message = message
        self.data = data

        super().__init__(message)


class NotFoundException(AppException):

    def __init__(
        self,
        message: str = "资源不存在",
        data: Any = None
    ):
        super().__init__(
            status_code=404,
            message=message,
            data=data
        )


class BadRequestException(AppException):

    def __init__(
        self,
        message: str = "请求参数错误",
        data: Any = None
    ):
        super().__init__(
            status_code=400,
            message=message,
            data=data
        )
