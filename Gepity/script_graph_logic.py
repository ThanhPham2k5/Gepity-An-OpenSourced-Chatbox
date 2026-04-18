from core import Graph_engine
from core.processor import split_docs_into_chunks
from langchain_community.document_loaders import PyPDFLoader
import time

def test_extraction(engine):
    file_paths = [
        "../LAB_01_Software Requirements Specification.pdf",
        "../project_report_final.pdf"
    ]   
    docs_with_chunks = {}

    print("--- Đang load và phân mảnh file ---")
    for path in file_paths:
        loader = PyPDFLoader(path)
        pages = loader.load() # pages là một list các Document
        
        if not pages:
            continue
            
        # Lấy source từ metadata của trang đầu tiên
        source_name = pages[0].metadata.get('source', path)
        
        # Phân mảnh toàn bộ các trang của file này
        chunks = split_docs_into_chunks(pages, chunk_size=800, chunk_overlap=50)
        # if source_name not in docs_with_chunks:
        #     docs_with_chunks[source_name] = []
        # docs_with_chunks[source_name].extend(chunks)
        docs_with_chunks[source_name] = chunks
    
    # Run the high-speed pipeline
    print("--- Đang bắt đầu sync ---")
    start_time = time.time()
    
    for filename, chunks in docs_with_chunks.items():
        print(f"Đang xử lý tài liệu: {filename}")

        engine.sync_to_graph(chunks, source=filename)
    # engine.sync_to_graph(chunks)
    
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
            response, sources = engine.get_response(user_input, "../project_report_final.pdf")
            
            end_time = time.time()
            duration = end_time - start_time

            # In kết quả
            print("-" * 30)
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

    # test_extraction(engine)

    test_response(engine)