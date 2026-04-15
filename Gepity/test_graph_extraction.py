from core import Graph_engine
from core.processor import split_docs_into_chunks
from langchain_community.document_loaders import PyPDFLoader

def test_run():
    # Initialize your engine
    engine = Graph_engine()
    
    # Load test file
    print("--- Đang load file ---")
    loader = PyPDFLoader("../proceduralSceneGeneration.pdf")
    documents = loader.load()

    # Get the chunks
    chunks = split_docs_into_chunks(documents, chunk_size=400, chunk_overlap=40)
    print(f"Số lượng chunks: {len(chunks)}")
    
    # Run the high-speed pipeline
    print("--- Đang bắt đầu sync ---")
    import time
    start_time = time.time()
    
    engine.sync_to_graph(chunks)
    
    end_time = time.time()
    print(f"--- Hoàn thành trong {end_time - start_time:.2f} giây ---")

if __name__ == "__main__":
    test_run()