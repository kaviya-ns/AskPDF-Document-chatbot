import pinecone 
import pdfplumber 
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext 
from llama_index.core import Document
from IPython.display import Markdown, display
from typing import List, Optional
import nltk
import os
import torch
import streamlit as st
import re
import io
from rouge_score import rouge_scorer
import pytesseract
from pdf2image import convert_from_bytes

from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)
import creds
# Initialize Pinecone
pc = Pinecone(api_key=creds.api_key)
index_name = "multi-doc-sum"
index = pc.Index(index_name)

import os
os.environ["HUGGINGFACE_HUB_TOKEN"] = str(creds.HF_TOKEN)

SUMMARY_INTENTS = [
    "summarize this document",
    "give me a summary",
    "what is this document about",
    "summarize the text",
    "provide an overview",
    "give me the main points",
    "what are the key takeaways",
    "tldr",
    "brief me on this",
    "what does this document say",
    "condense this",
    "give me a synopsis",
    "summarize all documents",
    "what is the gist of this",
    "can you summarize",
    "paraphrase this",
    "paraphrase the document",
    "rephrase this",
    "reword this",
    "give me a paraphrase",
    "explain this document",
    "what is this about",
    "overview of this"
]

@st.cache_resource
def load_embed_model():
    return HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        cache_folder=os.path.expanduser("~/.cache/huggingface/hub")
    )
@st.cache_resource
def load_intent_embeddings():
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    intent_vecs = model.encode(SUMMARY_INTENTS, convert_to_tensor=True)
    return model, intent_vecs, len(SUMMARY_INTENTS)

@st.cache_resource
def load_llm_summary():
    return Ollama(model="llama3.2:1b", request_timeout=300.0, context_window=4096)
    
embed_model = load_embed_model()
llm_summary = load_llm_summary()
intent_model, intent_vecs,_ = load_intent_embeddings()

def compute_rouge(reference: str, hypothesis: str) -> dict:
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores
    
# Streamlit UI setup
st.title("AskPDF~ Document Chatbot")
st.sidebar.header("Upload and Query PDF Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    return re.sub(r"[^\w\s]", "", text)

def extract_text_from_pdf(file):
    file_bytes = file.read()
    
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            width = page.width
            height = page.height
            left_bbox = (0, 50, width/2, height - 50)
            right_bbox = (width/2, 50, width, height - 50)
            left_text = page.crop(left_bbox).extract_text() or ""
            right_text = page.crop(right_bbox).extract_text() or ""

            if len(left_text.strip()) > 100 and len(right_text.strip()) > 100:
                text += left_text + "\n" + right_text + "\n"
            else:
                cropped = page.crop((0, 50, width, height - 50))
                text += cropped.extract_text() or ""
                text += "\n"

    if len(text.strip()) < 50:
        st.info(f"Scanned PDF detected, running OCR, this may take a moment...")
        images = convert_from_bytes(file_bytes, dpi=300)
        text = ""
        for i, image in enumerate(images):
            text += pytesseract.image_to_string(image) + "\n"

    return text
    
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences
    
def is_summary_intent(query: str, threshold: float = 0.45) -> bool:
    from torch.nn.functional import cosine_similarity
    qa_override_keywords = [
        "who", "when", "where", "how many", "what year", "which",
        "author", "name", "date", "cite", "published", "written by",
        "list the", "what is the", "what are the", "define", "explain what"
    ]
    if any(kw in query.lower() for kw in qa_override_keywords):
        return False  

    summary_keywords = ["paraphrase", "rephrase", "reword", "summarize", 
                        "summary", "gist", "tldr", "condense", "synopsis"]
    if any(kw in query.lower() for kw in summary_keywords):
        return True
    
    q_vec = intent_model.encode(query, convert_to_tensor=True).unsqueeze(0)
    sims = cosine_similarity(q_vec, intent_vecs)
    return sims.max().item() > threshold
                
if uploaded_files:
    try:
        index.delete(delete_all=True)
    except Exception:
        pass
    combined_text = ""
    document_texts = {}
    documents = []
    summaries_dict = {}
    for file in uploaded_files:
        document_text = extract_text_from_pdf(file)
        preprocessed_text = preprocess_text(document_text)
        document_texts[file.name] = preprocessed_text  
    
        document = Document(text=preprocessed_text)
        documents.append(document)  

        combined_text += preprocessed_text + "\n"  
    vector_store=PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    qa_prompt = PromptTemplate(
                "You are an assistant that answers ONLY from the provided context.\n"
                "Context:\n{context_str}\n"
                "Question: {query_str}\n"
                "If the answer is not in the context, say 'Not found in the document.'\n"
                "Answer:"
            )
    query_engine = vector_index.as_query_engine(llm=llm_summary, similarity_top_k=5, text_qa_template=qa_prompt)
    user_query = st.text_input("Ask a question from the given documents:")

    if st.button("Submit"):
        if is_summary_intent(user_query):
            for doc_name, text in document_texts.items():
                if len(text) > 3000:
                    st.info("Long document detected, using semantic chunking for better accuracy...")

                    document_obj = Document(text=text)
                    config = LanguageConfig(language="english", spacy_model="en_core_web_md")
                    splitter = SemanticDoubleMergingSplitterNodeParser(
                        language_config=config,
                        initial_threshold=0.4,
                        appending_threshold=0.5,
                        merging_threshold=0.5,
                        max_chunk_size=1000
                    )
                    nodes = splitter.get_nodes_from_documents([document_obj])
                    chunk_summaries = []

                    for node in nodes:
                        chunk_text = node.get_content().strip()
                        if not chunk_text:
                            continue
                        prompt = (
                            f"Summarize the following passage in 2-3 sentences:\n\n"
                            f"{chunk_text}\n\nSummary:"
                        )
                        chunk_summaries.append(str(llm_summary.complete(prompt)).strip())

                    combined = " ".join(chunk_summaries)[:3000]
                    prompt = (
                        f"Combine these partial summaries into one coherent 5-7 sentence summary:\n\n"
                        f"{combined}\n\nFinal Summary:"
                    )
                    full_summary_text = str(llm_summary.complete(prompt)).strip()

                else:
                    prompt = (
                        f"Read the document below and write a clear, concise summary "
                        f"in 5-7 sentences covering the main points.\n\n"
                        f"Document:\n{text}\n\nSummary:"
                    )
                    full_summary_text = str(llm_summary.complete(prompt)).strip()

                sentences = split_into_sentences(full_summary_text)
                st.write(f"**Summary for {doc_name}:**")
                for line in sentences:
                    if line.strip():
                        st.write(line)

                st.write("ROUGE Scores:")
                rouge_scores = compute_rouge(
                    document_texts[doc_name],
                    full_summary_text
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ROUGE-1", f"{rouge_scores['rouge1'].fmeasure:.3f}", help="Unigram overlap")
                with col2:
                    st.metric("ROUGE-2", f"{rouge_scores['rouge2'].fmeasure:.3f}", help="Bigram overlap")
                with col3:
                    st.metric("ROUGE-L", f"{rouge_scores['rougeL'].fmeasure:.3f}", help="Longest common subsequence")

        else:
            with st.spinner("Searching documents..."):
                response = query_engine.query(user_query)
                answer_text = str(response).strip()
                if answer_text:
                    st.write(f"Answer: {answer_text}")
                else:
                    st.write("Answer is not available in the given documents.")
