install:
	pip install -r requirements.txt

run:
	streamlit run app.py

pull-model:
	ollama pull qwen2.5:7b

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "faiss_index" -exec rm -rf {} +