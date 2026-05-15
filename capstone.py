import pinecone 
import pdfplumber 
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import PromptTemplate
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
import pytesseract
from pdf2image import convert_from_bytes
from transformers import pipeline
from llama_index.llms.groq import Groq

from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)
import creds
import subprocess


# Initialize Pinecone
pc = Pinecone(api_key=creds.api_key)
index_name = "multi-doc-sum"
index = pc.Index(index_name)


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
    return Groq(
        model="llama-3.1-8b-instant",
        api_key=creds.GROQ_API_KEY,  
        temperature=0.0,   
        max_tokens=512
    )
    
embed_model = load_embed_model()
llm_summary = load_llm_summary()
intent_model, intent_vecs,_ = load_intent_embeddings()

#Metrics
def load_nli_model():
    try:
        return pipeline(
            "text-classification",
            model="/home/kaviya/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-small/snapshots/fa2804872c3b4bd748f38c0185cc85775361e735",
            device=-1
        )
    except Exception as e:
        print(f"NLI model load failed: {e}")
        return None
    
def is_noise_chunk(text):
    words = text.split()
    if len(words) < 20:
        return True
    
    special = sum(1 for w in words if re.match(r'[\[\d]', w))
    if (special / len(words)) > 0.25:
        return True
    
    ref_signals = ["arXiv", "ar Xiv", "CoRR", "abs/", "preprint",
                   "Conference on", "Proceedings", "et al."]
    hits = sum(1 for s in ref_signals if s.lower() in text.lower())
    if hits >= 2:
        return True
    
    long_words = [w for w in words if len(w) > 15]
    if len(long_words) / len(words) > 0.3:
        return True
    
    return False

def get_source_context(source: str) -> str:
    clean = re.sub(r'\s+', ' ', source).strip()
    total = len(clean)
    if total < 1500:
        return clean
    beginning = clean[300:700]
    middle = clean[total//2 - 200: total//2 + 200]
    end = clean[-400:]
    return f"{beginning} {middle} {end}"

def compute_faithfulness(source: str, summary: str, nli_model) -> float:
    #Compares source context and summary to see how faithfull summary is to the context
    source_context = get_source_context(source)
    source_words = set(source.lower().split())

    sentences = [
        s.strip() for s in re.split(r'(?<=[.!?]) +', summary)
        if len(s.strip()) > 30
    ]
    if not sentences:
        return 0.0

    score = 0.0
    for sentence in sentences:
        result = nli_model(
            f"{source_context} [SEP] {sentence}",
            truncation=True, max_length=512
        )
        label = result[0]["label"].upper()
        conf = result[0]["score"]

        sent_words = set(sentence.lower().split()) - stop_words
        overlap = len(sent_words & source_words) / len(sent_words) if sent_words else 0.0

        if label == "ENTAILMENT":
            nli_score = 1.0
        elif label == "NEUTRAL":
            nli_score = 0.5
        else:
            nli_score = 0.0

        combined = 0.6 * nli_score + 0.4 * overlap
        score += combined

    final = score / len(sentences)
    return final

def compute_extractiveness(source: str, summary: str) -> dict:
    #Coverage: how much of summary content appears in source.
    #Density: how extractive vs abstractive the summary is.

    source_words = set(source.lower().split()) - stop_words
    summary_words = summary.lower().split()
    summary_content = [w for w in summary_words if w not in stop_words]
    
    if not summary_content:
        return {"coverage": 0.0, "abstractiveness": 0.0}
    
    covered = sum(1 for w in summary_content if w in source_words)
    coverage = covered / len(summary_content)
    abstractiveness = 1 - coverage  
    
    return {"coverage": coverage, "abstractiveness": abstractiveness}

# Streamlit UI setup
st.title("AskPDF~ Document Chatbot")
st.sidebar.header("Upload and Query PDF Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

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

            if (len(left_text.strip()) > 300 and len(right_text.strip()) > 300 and "." in left_text and "." in right_text):
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
    query_engine = vector_index.as_query_engine(llm=llm_summary, similarity_top_k=3, text_qa_template=qa_prompt)
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
                        max_chunk_size=300
                    )
                    nodes = splitter.get_nodes_from_documents([document_obj])
                    chunk_texts = [
                        node.get_content().strip() for node in nodes
                        if len(node.get_content().strip()) > 100
                        and not is_noise_chunk(node.get_content().strip())
                    ]
                    top_chunks = sorted(chunk_texts, key=len, reverse=True)[:3]

                    chunk_summaries = []
                    for chunk_text in top_chunks:
                        prompt = (
                            f"Summarize in 1-2 sentences using ONLY the text below. No outside knowledge.\n\n"
                            f"{chunk_text[:600]}\n\nSummary:"
                        )
                        chunk_summaries.append(str(llm_summary.complete(prompt)).strip())

                    combined = " ".join(chunk_summaries)[:1000]
                    prompt = (
                        f"Combine these into a coherent 5 sentence summary. Use only the information given:\n\n"
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
                nli_model = load_nli_model()

                if nli_model is not None:
                    faithfulness = compute_faithfulness(document_texts[doc_name], full_summary_text, nli_model)
                    extractiveness = compute_extractiveness(document_texts[doc_name], full_summary_text)

                    print(f"\n[Summary Metrics] {doc_name}")
                    print(f"  Faithfulness  : {faithfulness:.3f}")
                    print(f"  Coverage      : {extractiveness['coverage']:.3f}")
                    print(f"  Abstractiveness: {extractiveness['abstractiveness']:.3f}")
                else:
                    print(f"\n[Summary Metrics] Skipped: NLI model unavailable")
                                

        else:
            with st.spinner("Searching documents..."):
                response = query_engine.query(user_query)
                answer_text = str(response).strip()
                if answer_text:
                    st.write(f"Answer: {answer_text}")
                else:
                    st.write("Answer is not available in the given documents.")
