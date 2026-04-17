import sys
import os
import json
import torch

# core path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine import RAG_engine
from sentence_transformers import SentenceTransformer, util

# ground truth for llm checking
TEST_CASES = [
    {
        "id": "TC_01_OUT_OF_CONTEXT",
        "category": "Robustness",
        "question": "Thuật toán RAG hoạt động như thế nào?",
        "ground_truth": "Thông tin này không có trong tài liệu.",
        "keywords": ["không có", "không tìm thấy", "không phù hợp"],
        "should_contain_any": True
    },
    {
        "id": "TC_02_STAKEHOLDERS",
        "category": "Accuracy",
        "question": "Hệ thống E-Learn Plus có những stakeholders chính nào?",
        "ground_truth": "Các stakeholders chính bao gồm: Học sinh, Giảng viên, Quản trị viên và Nhân viên hỗ trợ.",
        "keywords": ["Học sinh", "Giảng viên", "Quản trị", "Phụ huynh", "Student", "Teacher", "Admin"],
        "should_contain_any": True
    }
]

eval_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def evaluate_hybrid(response: str, test_case: dict, threshold: float = 0.65) -> dict:
    response_lower = response.lower()
    
    matched = [kw for kw in test_case["keywords"] if kw.lower() in response_lower]
    kw_score = len(matched) / len(test_case["keywords"])
    kw_passed = kw_score > 0 if test_case["should_contain_any"] else kw_score == 1.0

    emb1 = eval_model.encode(response, convert_to_tensor=True)
    emb2 = eval_model.encode(test_case["ground_truth"], convert_to_tensor=True)
    semantic_score = util.cos_sim(emb1, emb2).item()
    semantic_passed = semantic_score >= threshold

    final_passed = semantic_passed or (kw_score >= 0.5)

    return {
        "passed": final_passed,
        "semantic_score": round(semantic_score, 2),
        "keyword_score": round(kw_score, 2),
        "matched_keywords": matched,
        "semantic_passed": semantic_passed,
        "keyword_passed": kw_passed
    }

def run_tests(engine, retriever, engine_name="RAG"):
    results = []
    os.makedirs("results", exist_ok=True)
    
    for i, test in enumerate(TEST_CASES):
        print(f"\n[{engine_name}] Test {i+1}: {test['question']}")
        
        response, docs = engine.get_response(
            user_input=test["question"],
            retriever=retriever
        )
        
        eval_result = evaluate_hybrid(response, test)
        
        results.append({
            "question": test["question"],
            "response": response,
            "evaluation": eval_result
        })
        
        status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
        print(f"  Status: {status} (Semantic: {eval_result['semantic_score']}, KW: {eval_result['keyword_score']})")
        print(f"  Matched: {eval_result['matched_keywords']}")
        print(f"  Response: {response[:150]}...")
    
    return results

if __name__ == "__main__":
    engine = RAG_engine()
    
    pdf_path = "data/LAB_01_Software Requirements Specification.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Lỗi: Không tìm thấy file tại {pdf_path}")
        sys.exit(1)

    print("--- Đang xử lý tài liệu ---")
    with open(pdf_path, "rb") as f:
        retriever, _, _, _ = engine.process_document([f])
    
    print("--- Đang chạy Test Cases ---")
    rag_results = run_tests(engine, retriever, "RAG")
    
    output_path = "results/test_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rag_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Hoàn thành! Kết quả đã lưu vào {output_path}")