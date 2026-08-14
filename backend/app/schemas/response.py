from typing import Generic, TypeVar

from pydantic import BaseModel


T = TypeVar("T")


class ResponseModel(BaseModel, Generic[T]):# 响应模型
    code: int = 200
    message: str = "success"
    data: T | None = None