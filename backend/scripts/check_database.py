from app.db.database import engine


try:
    connection = engine.connect()

    print("数据库连接成功")

    connection.close()

except Exception as e:
    print("数据库连接失败")
    print(e)
