from datetime import datetime

from pydantic import BaseModel


class DocumentResponse(BaseModel):

    id: int

    filename: str

    file_type: str | None

    file_size: int | None

    status: str

    created_at: datetime

    
    class Config:
        from_attributes = True #告诉Pydantic：可以读取对象属性，而不是只读取字典