import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent
env_path = ROOT_DIR / '.env'
load_dotenv(dotenv_path=env_path)

def test_neo4j_connection():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME", "neo4j") # Default to neo4j
    pwd = os.getenv("NEO4J_PASSWORD")
    
    print(f"--- Đang thử kết nối tới: {uri} ---")
    
    try:
        # We use database="neo4j" for Desktop
        graph = Neo4jGraph(
            url=uri, 
            username=user, 
            password=pwd, 
            database="neo4j",
            refresh_schema=True
        )
        
        # Test Query: Get a simple result from the DB
        result = graph.query("RETURN 'Kết nối thành công!' AS message")
        
        print("✅ Thành công!")
        print(f"Tin nhắn: {result[0]['message']}")
        
        # Check if GDS is installed (since you need it for your project)
        gds_check = graph.query("RETURN gds.version() AS gds_version")
        if gds_check:
            print(f"🚀 GDS Plugin: Đã cài đặt (Version: {gds_check[0]['gds_version']})")
        
    except Exception as e:
        print("❌ Kết nối thất bại.")
        print(f"Lỗi chi tiết: {e}")
        
        if "Unauthorized" in str(e):
            print("\n💡 Gợi ý: Kiểm tra lại Password trong file .env")
        elif "ServiceUnavailable" in str(e):
            print("\n💡 Gợi ý: Đảm bảo DBMS trong Neo4j Desktop đã được nhấn 'START'")

if __name__ == "__main__":
    test_neo4j_connection()