# 🤖 Gepity — An Open-Sourced Chatbox

> Upload a PDF or DOCX file and ask questions about its content — powered by local LLMs via Ollama.

---

## ✨ Features

- 📄 Upload **PDF / DOCX** documents
- 💬 Ask questions in **Vietnamese or English**
- 🔍 Semantic search with **FAISS** vector store
- 🧠 Runs fully **local** with **Qwen2.5:7b** via Ollama — no API key needed
- ⚡ Built with **Streamlit** for a fast, clean UI

---

## 🚀 Getting Started

### Prerequisites

- [WSL (Ubuntu)](https://learn.microsoft.com/en-us/windows/wsl/install)
- Python 3.8+
- [Ollama](https://ollama.ai) installed on your machine

---

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/ThanhPham2k5/Gepity-An-OpenSourced-Chatbox.git
cd Gepity-An-OpenSourced-Chatbox
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv myenv
source myenv/bin/activate
```

**3. Install dependencies**

```bash
cd Gepity/
make install
```

**4. Pull the LLM model** _(requires Ollama to be installed)_

```bash
make pull-model
```

---

### Running the App

```bash
make run
```

Then open your browser at `http://localhost:8501`

---

## 🗂️ Project Structure

```
Gepity-An-OpenSourced-Chatbox/
├── Gepity/
│   ├── app.py              # Main Streamlit application
│   ├── Makefile            # Shortcuts for common commands
│   └── requirements.txt    # Python dependencies
├── myenv/                  # Virtual environment (not committed)
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🛠️ Makefile Commands

| Command           | Description                      |
| ----------------- | -------------------------------- |
| `make install`    | Install Python dependencies      |
| `make run`        | Start the Streamlit app          |
| `make pull-model` | Download Qwen2.5:7b via Ollama   |
| `make clean`      | Remove cache and generated files |

---

## 📚 Tech Stack

| Layer            | Technology                                 |
| ---------------- | ------------------------------------------ |
| Frontend         | Streamlit                                  |
| LLM Runtime      | Ollama + Qwen2.5:7b                        |
| Embeddings       | sentence-transformers (multilingual MPNet) |
| Vector Store     | FAISS                                      |
| Document Parsing | PDFPlumber, PyPDF                          |
| Framework        | LangChain                                  |

---

## 📝 License

This project is licensed under the terms of the [LICENSE](./LICENSE) file.
