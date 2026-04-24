import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Tìm đường dẫn đến file .env ở thư mục gốc (Gepity-An-OpenSourced-Chatbox)
# __file__ là database.py -> parent là Gepity/ -> parent tiếp theo là thư mục gốc
ROOT_DIR = Path(__file__).resolve().parent.parent
env_path = ROOT_DIR / '.env'
load_dotenv(dotenv_path=env_path)

def get_graph_connection() ->Neo4jGraph | None:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    pwd = os.getenv("NEO4J_PASSWORD")
    name = os.getenv("NEO4J_DATABASENAME")
    
    try:
        graph = Neo4jGraph(
            url=uri, 
            username=user, 
            password=pwd, 
            database=name,
            refresh_schema=True
        )
        return graph
    except Exception as e:
        print(f"Lỗi Driver: {e}")
        return None
    