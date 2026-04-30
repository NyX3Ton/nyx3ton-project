import os, pyodbc
from dotenv import load_dotenv
load_dotenv(dotenv_path=r"D:\nemotron_mcp\.env")

cs = (
    f"Driver={{ODBC Driver 18 for SQL Server}};"
    f"Server={os.getenv('SQL_HOST')},{os.getenv('SQL_PORT','1433')};"
    f"Database={os.getenv('SQL_DATABASE')};"
    f"Uid={os.getenv('SQL_USER')};"
    f"Pwd={os.getenv('SQL_PASSWORD')};"
    f"Encrypt=no;TrustServerCertificate=no;"
)
print("CS:", cs)
cn = pyodbc.connect(cs, timeout=5)
cur = cn.cursor()
cur.execute("SELECT 1")
print("DB OK:", cur.fetchone()[0])
