VENV = ../myenv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

install:
	$(PIP) install -r requirements.txt

run:
	$(VENV)/bin/streamlit run app.py

pull-model:
	ollama pull qwen2.5:3b
	ollama pull qwen2.5:7b

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +