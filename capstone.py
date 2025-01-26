import pinecone 
import pdfplumber 
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.vector_stores.pinecone import PineconeVectorStore
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
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)
from dotenv import load_dotenv

load_dotenv()

# Access API keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index_name = "multi-doc-sum"
index = pc.Index(index_name)

# Load summarization model
checkpoint = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Initialize the HuggingFace LLM
os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
llm = HuggingFaceInferenceAPI(model_name="meta-llama/Llama-3.2-1B-Instruct", token=huggingface_api_key)

# Streamlit UI setup
st.title("AskPDF~ Document Chatbot")
st.sidebar.header("Upload and Query PDF Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    return re.sub(r"[^\w\s]", "", text)

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text
    return text

# Split text into sentences for line-by-line printing
def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return sentences

if uploaded_files:
    combined_text = ""
    document_texts = {}
    documents = []
    summaries_dict = {}
    for file in uploaded_files:
        #st.write(f"Processing file: {file.name}")
        document_text = extract_text_from_pdf(file)
        preprocessed_text = preprocess_text(document_text)
        document_texts[file.name] = preprocessed_text  # Store preprocessed text for each document
    
        # Create a Document object and add it to the documents list
        document = Document(text=preprocessed_text)
        documents.append(document)  # Append to the documents list

        combined_text += preprocessed_text + "\n"  # Combine all documents' text
    vector_store=PineconeVectorStore(pinecone_index=index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vector_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )

    # Single input box for user query
    user_query = st.text_input("Ask a question from the given documents:")

    if st.button("Submit"):
        # Check if the user query indicates a request for a summary
        summary_keywords = ["summarize", "summary", "generate a summary", "give a summary"]
        if any(keyword in user_query.lower() for keyword in summary_keywords):
            batch_size = 2
            config = LanguageConfig(language="english", spacy_model="en_core_web_md")
            splitter = SemanticDoubleMergingSplitterNodeParser(
                language_config=config,
                initial_threshold=0.4,
                appending_threshold=0.5,
                merging_threshold=0.5,
                max_chunk_size=500
            )

            for doc_name, text in document_texts.items():
                document_obj = Document(text=text)
                nodes = splitter.get_nodes_from_documents([document_obj])
                node_texts = [node.get_content().strip() for node in nodes if node.get_content().strip()]

                tokenized_inputs = tokenizer(node_texts, return_tensors="pt", truncation=True, padding=True, max_length=1024)
                input_ids, attention_mask = tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"]
                input_ids, attention_mask = input_ids.to(model.device), attention_mask.to(model.device)

                initial_summaries = []
                for i in range(0, len(input_ids), batch_size):
                    batch_input_ids = input_ids[i:i+batch_size]
                    batch_attention_mask = attention_mask[i:i+batch_size]
                    with torch.no_grad():
                        output = model.generate(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            max_length=400,
                            min_length=100,
                            top_k=50,
                            top_p=0.95,
                            temperature=0.7,
                            early_stopping=True
                        )
                    initial_summaries.extend([tokenizer.decode(out, skip_special_tokens=True) for out in output])

                concatenated_summary = " ".join(initial_summaries)
                if len(concatenated_summary) > 1024:
                    concatenated_summary = concatenated_summary[:1024]

                tokenized_summary = tokenizer(concatenated_summary, return_tensors="pt", truncation=True, padding=True, max_length=1024)
                summary_output = model.generate(
                    input_ids=tokenized_summary["input_ids"].to(model.device),
                    attention_mask=tokenized_summary["attention_mask"].to(model.device),
                    max_length=600,
                    min_length=200,
                    top_k=50,
                    top_p=0.95,
                    temperature=0.7,
                    early_stopping=True
                )

                full_summary = tokenizer.decode(summary_output[0], skip_special_tokens=True)
                sentences = split_into_sentences(full_summary)
                summaries_dict[doc_name] = sentences

            for doc_name, summary in summaries_dict.items():
                st.write(f"\nSummary for {doc_name}:\n")
                for line in summary[:5]:  # Display only the first 5 sentences for testing
                    st.write(line)

        else:
            # Query using the LLM
            query_engine = vector_index.as_query_engine(llm=llm)
            response = query_engine.query(user_query)

             # Access the content of the response (adjust 'response.text' based on the actual structure)
            if hasattr(response, 'text'):
                answer_text = response.text.strip()  # Access the text and strip whitespace
            else:
                answer_text = str(response).strip()  # Fallback to string representation if text attribute is not present

            if answer_text:
                st.write(f"Answer: {answer_text}")
            else:
                st.write("Answer is not available in the given documents.")

