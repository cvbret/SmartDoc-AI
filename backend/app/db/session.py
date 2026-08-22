from app.db.database import SessionLocal


def get_db():

    db = SessionLocal()

    try:
        yield db#返回数据库会话对象

    finally:
        db.close()
