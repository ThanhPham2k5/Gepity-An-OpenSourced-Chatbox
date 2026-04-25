import os
import time
import pandas as pd
from pathlib import Path
from io import BytesIO

# --- DeepEval Imports ---
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric
)
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

# --- 1. Cấu hình Giám khảo (Judge) Groq API ---
class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        # Nghỉ 3 giây để tránh chạm ngưỡng TPM (6000 token/phút rất dễ bị đầy)
        time.sleep(20) 
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional auditor. Always output valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model=self.model_name,
            response_format={"type": "json_object"}
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# --- 2. Import GraphRAG Components ---
try:
    from core.graph_rag_engine import Graph_engine
except ImportError:
    print("[LỖI] Cần file core/graph_rag_engine.py để chạy.")
    exit(1)

class FakeUploadedFile:
    def __init__(self, file_path: str):
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy: {p.resolve()}")
        self.name, self.size = p.name, p.stat().st_size
        self._buf = BytesIO(p.read_bytes())
    def read(self, n=-1): return self._buf.read(n) if n != -1 else self._buf.read()
    def seek(self, pos): self._buf.seek(pos)
    def tell(self): return self._buf.tell()

# --- 3. Main Execution ---
def run_graph_evaluation():
    # Mark: Bắt đầu toàn bộ quy trình
    start_all = time.perf_counter()

    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1200"
    os.environ["DEEPEVAL_MAX_CONCURRENT_TEST_CASES"] = "1"

    # Cấu hình Model
    RESPONSE_MODEL = "qwen2.5:3b" 
    JUDGE_MODEL = "llama-3.3-70b-versatile"
    
    # Dùng 1 Key cố định
    GROQ_API_KEY = ""

    TEST_DOCUMENTS = ["../LAB_01_Software Requirements Specification.pdf"] 
    # CHUNK_SIZE = 500
    # CHUNK_OVERLAP = 50
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 80
    # CHUNK_SIZE = 1000
    # CHUNK_OVERLAP = 100
    # CHUNK_SIZE = 500
    # CHUNK_OVERLAP = 100

    TEST_CASES = [
        {
            "question": "Hệ thống E-Learn Plus hỗ trợ những vai trò người dùng chính nào?",
            "ground_truth": "Học sinh, Giảng viên và Quản trị viên."
        },
        {
            "question": "Yêu cầu số 7 về Quản lý Nội dung Đa phương tiện bao gồm những tính năng bảo vệ nội dung nào?",
            "ground_truth": "Watermark và bảo vệ nội dung."
        },
        {
            "question": "Làm thế nào hệ thống đảm bảo tốc độ tải nội dung đa phương tiện nhanh chóng?",
            "ground_truth": "Sử dụng CDN (Content Delivery Network)."
        },
        {
            "question": "Các định dạng tệp tin nào được hỗ trợ trong hệ thống khóa học?",
            "ground_truth": "Video, audio, PDF, và văn bản."
        },
        {
            "question": "Hệ thống sử dụng công nghệ gì để dự đoán nguy cơ bỏ học của học sinh?",
            "ground_truth": "Sử dụng AI/ML (Trí tuệ nhân tạo/Học máy)."
        },

        # {
        #     "question": "Liệt kê các loại câu hỏi mà giảng viên có thể tạo trong hệ thống bài tập.",
        #     "ground_truth": "Trắc nghiệm, tự luận, điền khuyết."
        # },
        # {
        #     "question": "Để đăng ký tài khoản, người dùng cần thực hiện bước xác thực nào?",
        #     "ground_truth": "Xác thực qua email."
        # },
        # {
        #     "question": "E-Learn Plus tuân thủ những quy định bảo mật và dữ liệu quốc tế nào?",
        #     "ground_truth": "Tuân thủ GDPR và các quy định về dữ liệu."
        # },
        # {
        #     "question": "Các tính năng nào thuộc về Tương thích Đa nền tảng?",
        #     "ground_truth": "Responsive design, tương thích trình duyệt chính, ứng dụng mobile native (iOS/Android), hoạt động offline cơ bản, đồng bộ đa thiết bị."
        # },
        # {
        #     "question": "Hệ thống báo cáo thông minh cung cấp những loại biểu đồ hay bảng điều khiển (dashboard) nào?",
        #     "ground_truth": "Dashboard analytics thời gian thực."
        # },

        # {
        #     "question": "Trong bài tập (Câu 10), Ma trận theo dõi yêu cầu cần bao gồm những trường thông tin nào?",
        #     "ground_truth": "ID yêu cầu, Tên yêu cầu, Mức độ ưu tiên, Trạng thái, và Ghi chú."
        # },
        # {
        #     "question": "Có những phương thức giao tiếp nào giữa học sinh và giảng viên được đề cập?",
        #     "ground_truth": "Diễn đàn thảo luận, tin nhắn riêng tư, bình luận bài học, thông báo thời gian thực, video conference."
        # },
        # {
        #     "question": "Hệ thống thanh toán có hỗ trợ việc hoàn tiền không?",
        #     "ground_truth": "Có, yêu cầu số 6 đề cập đến Hoàn tiền và xử lý tranh chấp."
        # },
        # {
        #     "question": "Yêu cầu số 9 đề cập đến việc xác thực hai yếu tố như thế nào?",
        #     "ground_truth": "Xác thực hai yếu tố (2FA)."
        # },
        # {
        #     "question": "AI được ứng dụng như thế nào để cá nhân hóa việc học tập?",
        #     "ground_truth": "Phân tích hành vi học tập và đưa ra khuyến nghị khóa học cá nhân hóa."
        # },
    ]

    engine = Graph_engine(summary_model_name=RESPONSE_MODEL, response_model_name=RESPONSE_MODEL)
    source_name = Path(TEST_DOCUMENTS[0]).stem
    
    # --- Giai đoạn 1: Indexing ---
    # print(f"\n[1/3] Indexing vào Neo4j...")
    # files = [FakeUploadedFile(p) for p in TEST_DOCUMENTS]
    # docs_with_chunks = engine.process_document(files, CHUNK_SIZE, CHUNK_OVERLAP)
    # for filename, chunks in docs_with_chunks.items():
    #     engine.sync_to_graph(chunks, source=filename)
    # engine.create_vector_indexes()
    # engine.create_fulltext_index()


    # --- Giai đoạn 2: Inference ---
    print(f"\n[2/3] Chạy Inference...")
    test_cases_for_deepeval = []
    inference_latencies = []

    for idx, tc in enumerate(TEST_CASES, 1):
        q, gt = tc["question"], tc["ground_truth"]
        
        s_inf = time.perf_counter()
        # pure vector search
        # answer, raw_context_data = engine.get_response(user_input=q, doc_source=source_name, hybrid_search=False)
        # hybrid search
        answer, raw_context_data = engine.get_response(user_input=q, doc_source=source_name, hybrid_search=True)
        e_inf = time.perf_counter()

        inf_time = e_inf - s_inf
        inference_latencies.append({"Câu hỏi": q[:30]+"...", "Thời gian (s)": round(inf_time, 3)})

        # Giới hạn context để tiết kiệm token trên 1 API key duy nhất
        retrieval_context = []
        if isinstance(raw_context_data, list):
            for item in raw_context_data[:3]: 
                text = item.get('text', '') if isinstance(item, dict) else str(item)
                retrieval_context.append(text)
        
        if not retrieval_context:
            retrieval_context = ["No graph context retrieved."]

        test_cases_for_deepeval.append(LLMTestCase(
            input=q, actual_output=answer, retrieval_context=retrieval_context, expected_output=gt
        ))
        print(f"   ✅ Hoàn tất Inference câu {idx}")
        time.sleep(10) 

    # --- Giai đoạn 3: Evaluation ---
    print(f"\n[3/3] Đang chấm điểm với {JUDGE_MODEL}...")
    
    groq_judge = GroqJudge(model_name=JUDGE_MODEL, api_key=GROQ_API_KEY)
    metrics = [
        FaithfulnessMetric(threshold=0.5, model=groq_judge),
        AnswerRelevancyMetric(threshold=0.5, model=groq_judge),
        ContextualPrecisionMetric(threshold=0.4, model=groq_judge),
        ContextualRecallMetric(threshold=0.4, model=groq_judge)
    ]

    evaluate(test_cases_for_deepeval, metrics)

    print("\n[4/4] BÁO CÁO THỜI GIAN PHẢN HỒI")
    df_latency = pd.DataFrame(inference_latencies)
    print(df_latency.to_string(index=False))
    print(f"\n⏱️ Trung bình: {df_latency['Thời gian (s)'].mean():.3f} giây/câu")

    print("\n" + "=" * 65)
    print("   HOÀN TẤT ĐÁNH GIÁ GRAPHRAG")
    print("=" * 65)

if __name__ == "__main__":
    run_graph_evaluation()