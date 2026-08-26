import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base

from app.core.config import settings


logger = logging.getLogger(__name__)


#创建数据库连接入口
engine = create_engine(
    settings.database_url
)

logger.debug("Database engine initialized")

#创建数据库操作会话
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()
