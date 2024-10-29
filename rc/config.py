#!/usr/bin/env python3
import os
from pathlib import Path
import stanza
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# Directory Configuration
HTML_DIRECTORY = "html"
INDEX_DIRECTORY = "index-dir-html"

# Model Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama3.2"

# Initialize stanza
def init_stanza():
    model_dir = Path.home() / "stanza_resources" / "en"
    if not model_dir.exists():
        stanza.download('en')
    return stanza.Pipeline('en', processors='tokenize')

# Initialize models
def init_models():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    llm = ChatOllama(model=LLM_MODEL_NAME)
    nlp = init_stanza()
    return embeddings, llm, nlp

# Global variables after initialization
embeddings, llm, nlp = init_models()