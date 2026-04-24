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
# Sử dụng Base Class để tự định nghĩa model Groq
from deepeval.models.base_model import DeepEvalBaseLLM
from openai import OpenAI

class GroqJudge(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name
        # Khởi tạo client OpenAI trỏ về Groq
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key="" 
        )

    def load_model(self):
        """DeepEval yêu cầu phương thức này để trả về đối tượng model gốc"""
        return self.client

    def generate(self, prompt: str) -> str:
        """Phương thức đồng bộ để tạo phản hồi"""
        time.sleep(3) 
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that always outputs valid JSON."},
                {"role": "user", "content": prompt}
            ],
            model=self.model_name,
            # Ép Groq trả về JSON để tránh lỗi JSONDecodeError
            response_format={"type": "json_object"}
        )
        return chat_completion.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        """Phương thức bất đồng bộ (async) mà DeepEval thực sự gọi"""
        # Vì gọi API, ta có thể dùng generate thông thường bên trong
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# --- 2. Các Class hỗ trợ (Giữ nguyên logic của bạn) ---
class FakeUploadedFile:
    def __init__(self, file_path: str):
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy: {p.resolve()}")
        self.name, self.size = p.name, p.stat().st_size
        self._buf = BytesIO(p.read_bytes())

    def read(self, n=-1):
        return self._buf.read(n) if n != -1 else self._buf.read()

    def seek(self, pos):
        self._buf.seek(pos)

    def tell(self):
        return self._buf.tell()

try:
    from core.engine import RAG_engine
except ImportError:
    print("[LỖI] Cần file core/engine.py để chạy RAG.")
    exit(1)

# --- 3. Main Execution ---
def run_evaluation():
    os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = "1200"
    os.environ["DEEPEVAL_MAX_CONCURRENT_TEST_CASES"] = "1"

    MODEL_NAME = "qwen2.5:3b" # RAG chạy local
    JUDGE_MODEL_NAME = "llama-3.3-70b-versatile" # Judge chạy Groq API
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

    rag = RAG_engine(model_name=MODEL_NAME)
    retriever = None

    # vector_retriever = None
    if TEST_DOCUMENTS:
        files = [FakeUploadedFile(p) for p in TEST_DOCUMENTS]
        retriever, _, _, _ = rag.process_document(files, CHUNK_SIZE, CHUNK_OVERLAP)
        # _, vector_store, _, _ = rag.process_document(files, CHUNK_SIZE, CHUNK_OVERLAP)
        # vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    test_cases_for_deepeval = []
    latency_logs = []

    for idx, tc in enumerate(TEST_CASES, 1):
        q, gt = tc["question"], tc["ground_truth"]

        t1 = time.perf_counter()

        answer, docs = rag.get_response(user_input=q, retriever=retriever, chat_history=[])
        # answer, _ = rag.get_response(user_input=q, retriever=vector_retriever, chat_history=[])

        t3 = time.perf_counter()

        latency_logs.append({"question": q, "total": t3 - t1})
        retrieval_context = [doc.page_content for doc in docs] if docs else []
        test_cases_for_deepeval.append(LLMTestCase(
            input=q,
            actual_output=answer,
            retrieval_context=retrieval_context,
            expected_output=gt
        ))
        time.sleep(20)

    # Khởi tạo Judge Groq
    groq_judge = GroqJudge(model_name=JUDGE_MODEL_NAME)
    
    metrics = [
        FaithfulnessMetric(threshold=0.5, model=groq_judge),
        AnswerRelevancyMetric(threshold=0.5, model=groq_judge),
        ContextualPrecisionMetric(threshold=0.5, model=groq_judge),
        ContextualRecallMetric(threshold=0.5, model=groq_judge)
    ]

    evaluate(test_cases_for_deepeval, metrics)

    df_latency = pd.DataFrame(latency_logs)
    
    # Làm tròn số cho dễ nhìn (giây)

    df_latency['total'] = df_latency['total'].round(3)
    
    # In bảng ra console
    print(df_latency.to_string(index=False))

    # Tính trung bình cộng
    avg_total = df_latency['total'].mean()
    print(f"\n⏱️ Thời gian phản hồi trung bình: {avg_total:.3f} giây/câu")


    print("\n" + "=" * 62)
    print("   HOÀN TẤT ĐÁNH GIÁ")
    print("=" * 62)

if __name__ == "__main__":
    run_evaluation()