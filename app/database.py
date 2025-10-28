# app/database.py
import pyodbc

server = '43.200.45.240,1433'
database = 'mina'
username = 'sa'
password = 'mina123!'
driver = '{ODBC Driver 17 for SQL Server}'  # 중괄호 포함된 상태

connection_string = (
    f"DRIVER={driver};"        # ← 여긴 중괄호 중복 X
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