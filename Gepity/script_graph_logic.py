from core import Graph_engine
from core.processor import split_docs_into_chunks
from langchain_community.document_loaders import PyPDFLoader
import time

def test_extraction(engine):    
    # Load test file
    print("--- Đang load file ---")
    loader = PyPDFLoader("../LAB_01_Software Requirements Specification.pdf")
    documents = loader.load()

    # Get the chunks
    chunks = split_docs_into_chunks(documents, chunk_size=400, chunk_overlap=40)
    print(f"Số lượng chunks: {len(chunks)}")
    
    # Run the high-speed pipeline
    print("--- Đang bắt đầu sync ---")
    start_time = time.time()
    
    engine.sync_to_graph(chunks)
    
    end_time = time.time()
    print(f"--- Hoàn thành trong {end_time - start_time:.2f} giây ---")

def test_response(engine):
    print("\n" + "="*60)
    print("🤖 GEPI-GRAG: HỆ THỐNG TRUY VẤN ĐỒ THỊ TRI THỨC")
    print("Nhập 'exit', 'quit' hoặc 'q' để dừng chương trình.")
    print("="*60)

    while True:
        # Lấy input từ console
        user_input = input("\nĐặt câu hỏi cho dữ liệu của bạn: ").strip()

        # Kiểm tra điều kiện thoát
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\n👋 Tạm biệt! Hy vọng đồ thị đã giúp ích cho bạn.")
            break

        # Bỏ qua nếu input trống
        if not user_input:
            continue

        print("\nĐang truy vấn Graph + LLM...")
        
        try:
            # Đo thời gian phản hồi
            start_time = time.time()
            
            # Gọi hàm xử lý chính
            response_type, response, sources = engine.get_response(user_input)
            
            end_time = time.time()
            duration = end_time - start_time

            # In kết quả
            print("-" * 30)
            if(response_type):
                print(f"LOẠI CÂU HỎI: {response_type}")
            print(f"CÂU TRẢ LỜI ({duration:.2f} giây):")
            print(response)
            if sources:
                print("\n" + "CÁC ĐOẠN VĂN BẢN GỐC (THAM KHẢO):".center(40, "-"))
                for i, src in enumerate(sources, 1):
                    
                    score = src.get('score', 0)
                    print(f"[{i}] (Score: {score:.2f}):")
                    print(f"{src['text'].strip()}\n")
                    print(f"Page: {src['page_number']}\n")
            print("-" * 30)
            
        except Exception as e:
            print(f"Có lỗi xảy ra trong quá trình xử lý: {e}")

if __name__ == "__main__":
    # Initialize engine
    engine = Graph_engine()

    # test_extraction(engine)s

    test_response(engine)