from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime, timezone

from app.db.database import Base

#创建一个叫 Document 的 Python 类，它对应数据库中的一张表。
class Document(Base):

    __tablename__ = "documents" #数据库表名

    #Column 是 SQLAlchemy 提供的一个类（Class）
    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    #filename 是文档的文件名，它是一个字符串，不能为 NULL。
    filename = Column(
        String,
        nullable=False
    )

    #file_type 是文档的文件类型，它是一个字符串，可以为 NULL。
    file_type = Column(
        String
    )

    #file_size 是文档的文件大小，它是一个整数，可以为 NULL。
    file_size = Column(
        Integer
    )

    #status 是文档的状态，它是一个字符串，默认值为 "processing"。
    status = Column(
        String,
        default="processing"
    )

    #created_at 是文档的创建时间，它是一个 DateTime 类型，默认值为当前时间。
    created_at = Column(
    DateTime(timezone=True),
    default=lambda: datetime.now(timezone.utc)
    )