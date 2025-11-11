# app/database.py
import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()  # .env 파일 불러오기

server = os.getenv("DB_SERVER")
database = os.getenv("DB_NAME")
username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
driver = "{ODBC Driver 17 for SQL Server}"

connection_string = (
    f"DRIVER={driver};"
    f"SERVER={server};"
    f"DATABASE={database};"
    f"UID={username};"
    f"PWD={password};"
    "Encrypt=no;"
)

def get_connection():
    try:
        conn = pyodbc.connect(connection_string)
        print("데이터베이스 연결 성공")
        return conn
    except Exception as e:
        print("데이터베이스 연결 실패:", e)
        raise
