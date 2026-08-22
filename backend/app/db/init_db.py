from app.db.database import engine, Base

from app.models.document import Document


print("开始创建数据库表...")


Base.metadata.create_all(
    bind=engine
)


print("数据库表创建完成")