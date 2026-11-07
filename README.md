# AskPDF ~ Document Chatbot

A Retrieval-Augmented Generation (RAG) application that lets you upload PDF documents and ask questions or request summaries, using Groq (Llama 3.1 8B), LlamaIndex, and Pinecone.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-UI-red) ![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-green) ![Pinecone](https://img.shields.io/badge/Pinecone-VectorDB-purple)

---

## Features

- **Multi-document upload** : upload and query multiple PDFs simultaneously
- **Smart intent detection** : automatically routes between summarization and Q&A using semantic similarity
- **Two-column PDF support** : detects and handles academic paper layouts 
- **OCR fallback** : scanned PDFs are handled via Tesseract 
- **Semantic chunking** : uses `SemanticDoubleMergingSplitterNodeParser` for long documents
- **Noise filtering** : filters out garbled text, reference sections, and table artifacts before summarization
- **Faithfulness metrics** : NLI-based faithfulness scoring + lexical coverage/abstractiveness metrics
  
---

## Architecture

```
PDF Upload
    │
    ▼
Text Extraction (pdfplumber + OCR fallback)
    │
    ▼
Preprocessing & Chunking
    │
    ├──► Summarization path (SemanticDoubleMergingSplitter → noise filter → Groq)
    │
    └──► Q&A path (Pinecone vector store → similarity retrieval → Groq)
    │
    ▼
Faithfulness & Coverage Metrics (NLI + lexical overlap)
```

---

## Project Structure

```
AskPDF-Document-chatbot/
├── capstone.py         # Main Streamlit application
├── creds.py            # API keys (create this yourself, git-ignored)
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kaviya-ns/AskPDF-Document-chatbot.git
cd AskPDF-Document-chatbot
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

> ⚠️ **Important:** The NLI faithfulness metric uses `cross-encoder/nli-deberta-v3-small` via the `transformers` pipeline. This requires a specific version of `transformers` to load correctly. Install in this order to avoid conflicts:

```bash
pip install torch==2.1.0
pip install transformers==4.36.2
pip install sentence-transformers
pip install llama-index
pip install llama-index-embeddings-huggingface
pip install llama-index-vector-stores-pinecone
pip install llama-index-llms-groq
pip install llama-index-core
pip install pinecone-client
pip install pdfplumber
pip install pdf2image
pip install pytesseract
pip install streamlit
pip install spacy
python -m spacy download en_core_web_md
```

If you hit dependency conflicts between `llama-index` and `transformers`, pin `transformers==4.36.2`. this is the version the NLI model pipeline was validated against.

### 4. System dependencies (Ubuntu/Debian)

```bash
sudo apt install tesseract-ocr poppler-utils
```

### 5. Configure API keys

Create a `creds.py` file in the project root:

```python
# creds.py
api_key = "your-pinecone-api-key"
HF_TOKEN = "your-huggingface-token"
GROQ_API_KEY = "your-groq-api-key"
```

Get your keys from:
- Pinecone: [pinecone.io](https://www.pinecone.io)
- Groq (free): [console.groq.com](https://console.groq.com)
- HuggingFace: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Add `creds.py` to `.gitignore`:

```bash
echo "creds.py" >> .gitignore
```

### 6. Set up Pinecone index

Create an index named `multi-doc-sum` in your Pinecone dashboard with:
- **Dimensions:** 384 (matches `all-MiniLM-L6-v2`)
- **Metric:** cosine
- **Type:** Serverless

### 7. Download the NLI model

The faithfulness metric uses `cross-encoder/nli-deberta-v3-small`. It will download automatically on first run, or you can pre-download it:

```python
from transformers import pipeline
pipeline("text-classification", model="cross-encoder/nli-deberta-v3-small")
```

---

## Running the App

```bash
streamlit run capstone.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Metrics

| Metric | Description |
|---|---|
| **Faithfulness** | Combined NLI entailment score + lexical overlap (0–1, higher = more grounded) |
| **Coverage** | Fraction of summary vocabulary found in source document |
| **Abstractiveness** | `1 - coverage`; how much the summary paraphrases vs extracts |


---
