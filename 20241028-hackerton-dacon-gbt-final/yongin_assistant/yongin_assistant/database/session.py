import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

SQLITE_LOCAL_PATH = 'datasets/yongin.db'
DATABASE_URL = f'sqlite:///{SQLITE_LOCAL_PATH}'

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    from yongin_assistant.database.models import Base
    Base.metadata.create_all(bind=engine)

if not os.path.exists(SQLITE_LOCAL_PATH):
    init_db()
