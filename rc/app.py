import os
import time
import io
import json
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool
import fitz  # PyMuPDF for advanced PDF handling
import camelot  # For table extraction
import pytesseract  # For OCR on images and diagrams
from PIL import Image
from transformers import pipeline  # For summarization and intent analysis
import stanza
import numpy as np


stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize')

# Summarization and intent analysis models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Constants for directories and file paths
PDF_DIRECTORY = "data-pdfs"
INDEX_DIRECTORY = "index-dir"
METADATA_FILE = "index-dir/metadata.json"

# Define embedding model (HuggingFace model)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Function to load metadata
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

# Ensure the index directory exists
def ensure_index_directory():
    if not os.path.exists(INDEX_DIRECTORY):
        os.makedirs(INDEX_DIRECTORY)

# Function to save metadata
def save_metadata(metadata):
    ensure_index_directory()
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

# Function to load PDFs, split text, and embed the documents
def ingest_pdfs():
    st.write("Starting PDF ingestion...")
    st.write("Starting PDF ingestion...")
    metadata = load_metadata()
    documents = []
    st.write("Ingesting PDFs from the directory...")
    for file_name in os.listdir(PDF_DIRECTORY):
        if file_name.endswith(".pdf"):
            print(file_name)
            pdf_path = os.path.join(PDF_DIRECTORY, file_name)
            file_metadata = metadata.get(file_name, {})
            last_modified = os.path.getmtime(pdf_path)
            if file_metadata.get("last_modified") == last_modified:
                continue
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                st.write(f"Extracting from page {page_num + 1} of {file_name}...")
                page = doc.load_page(page_num)
                # Extract text from the page
                text = page.get_text()
                if text.strip():
                    documents.append(Document(page_content=text, metadata={"semantic_tag": "general_text"}))
                    print("Extracted text sample:")
                    print("\n".join(text.splitlines()[:15]))
                else:
                    st.write(f"No text found on page {page_num} of {file_name}")
                # Extract tables using Camelot
                tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1))
                if tables:
                    print(tables)
                    for table in tables:
                        table_text = table.df.to_string()
                        if len(table_text) > 1024:
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
                            chunks = text_splitter.split_text(table_text)
                            summaries = []
                            for chunk in chunks:
                                summary = summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
                                summaries.append(summary)
                            table_summary = ' '.join(summaries)
                        else:
                            table_summary = summarizer(table_text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
                        documents.append(Document(page_content=table_summary, metadata={"semantic_tag": "numerical_data"}))

                # Extract images and perform OCR
                images = page.get_images(full=True)
                for img in images:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(image)
                    print(ocr_text)
                    if ocr_text.strip():
                        if len(ocr_text) > 1024:
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
                            chunks = text_splitter.split_text(ocr_text)
                            summaries = []
                            for chunk in chunks:
                                summary = summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
                                summaries.append(summary)
                            ocr_summary = ' '.join(summaries)
                        else:
                            st.write(f"No OCR text found in image on page {page_num} of {file_name}")

            # Update metadata
            metadata[file_name] = {"last_modified": last_modified}

    # Split documents into chunks for easier processing
    texts = []
    for doc in documents:
        if len(doc.page_content.strip()) > 0:
            nlp_doc = nlp(doc.page_content)
            sentences = [sentence.text for sentence in nlp_doc.sentences]
            for sent in sentences:
                if len(sent.strip()) > 0:
                    texts.append(Document(page_content=sent.strip(), metadata=doc.metadata))

    # Use embeddings to embed chunks
    documents = [Document(page_content=t.page_content, metadata={"semantic_tag": t.metadata["semantic_tag"]}) for t in texts]
    if documents:
        st.write(f"Successfully ingested {len(documents)} documents.")
        if documents:
            vectorstore = FAISS.from_documents(documents, embeddings)
            print("Vectorstore successfully created.")
        else:
            st.write("No valid documents found to create a vectorstore.")
            vectorstore = None
    else:
        st.write("No valid documents found in the PDFs for embedding.")
        vectorstore = None
    save_metadata(metadata)
    return vectorstore, texts

# Function to update the vectorstore
def update_vectorstore(vectorstore):
    texts = []
    metadata = load_metadata()
    for file_name in os.listdir(PDF_DIRECTORY):
        pdf_path = os.path.join(PDF_DIRECTORY, file_name)
        file_metadata = metadata.get(file_name, {})
        last_modified = os.path.getmtime(pdf_path)
        # Check if the document is already indexed (by comparing metadata)
        if file_metadata.get("last_modified") == last_modified:
            continue
        doc = fitz.open(pdf_path)
        documents = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Extract text from the page
            text = page.get_text()
            if text.strip():
                documents.append({"type": "text", "content": text, "semantic_tag": "general_text"})
            # Extract tables using Camelot
            tables = camelot.read_pdf(pdf_path, pages=str(page_num + 1))
            for table in tables:
                table_summary = summarizer(table.df.to_string(), max_length=50, min_length=25, do_sample=False)[0]['summary_text']
                documents.append({"type": "table", "content": table.df.to_json(), "summary": table_summary, "semantic_tag": "numerical_data"})
            # Extract images and perform OCR
            images = page.get_images(full=True)
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                ocr_text = pytesseract.image_to_string(image)
                if ocr_text.strip():
                    ocr_summary = summarizer(ocr_text, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
                    documents.append({"type": "image_ocr", "content": ocr_text, "summary": ocr_summary, "semantic_tag": "image_extracted_text"})

        # Split documents into chunks for easier processing
        texts = []
        for doc in documents:
            if doc["type"] == "text":
                # Use Stanza to process the text and split by sentences
                nlp_doc = nlp(doc["content"])
                sentences = [sentence.text for sentence in nlp_doc.sentences]
                for sent in sentences:
                    if len(sent.strip()) > 0:
                        texts.append({"content": sent.strip(), "semantic_tag": doc["semantic_tag"]})
            else:
                # For tables and OCR, use summaries for chunking
                texts.append({"content": doc["summary"], "semantic_tag": doc["semantic_tag"]})

        new_documents = [Document(page_content=t["content"], metadata={"semantic_tag": t["semantic_tag"]}) for t in texts]
        if new_documents:
            vectorstore.add_documents(new_documents)
        else:
            st.write("No new valid documents to add to the FAISS index.")
        # Update metadata
        metadata[file_name] = {"last_modified": last_modified}
    save_metadata(metadata)
    return vectorstore, texts

# Ollama LLM setup
llm = ChatOllama(model="llama3.2")

# Define custom tools
# Tool for code generation (e.g., generating API call examples)
def generate_code_snippet(query):
    # Here, a simple mockup code generation
    return f"# Example API Call based on query: '{query}'\nimport requests\nresponse = requests.get('https://api.example.com/resource')\nprint(response.json())"

# Tool for extracting detailed API information
def extract_api_details(query):
    return "This is a mockup API detail extraction for query: " + query

# Define tools
tools = [
    Tool(
        name="Code Generator",
        func=generate_code_snippet,
        description="Generate code snippets related to the API documentation."
    ),
    Tool(
        name="API Detail Extractor",
        func=extract_api_details,
        description="Extract detailed API information such as parameters, endpoints, etc."
    )
]

# Initialize an agent with tools and ChatOllama model
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True, handle_parsing_errors=True)

# Setting up the retrieval-based QA chain
st.set_page_config(page_title="RAG Application using LangChain, Ollama, and Streamlit")

st.title("RAG Application with Langchain and Streamlit")
st.write("This app uses an RAG approach to answer questions from PDFs stored in a directory.")

# Initial vectorstore creation
st.sidebar.header("Ingestion Settings")
update_index = st.sidebar.button("Update Index")

if update_index:
    st.write("Updating vector store, please wait...")
    vectorstore, texts = ingest_pdfs()
    # Optionally save your index to disk to make it persistent
    ensure_index_directory()
    if vectorstore is not None:
        vectorstore.save_local(INDEX_DIRECTORY)
        st.write("Vector store updated!")
    else:
        st.write("Vectorstore could not be created due to lack of valid documents.")
elif os.path.exists(os.path.join(INDEX_DIRECTORY, 'index.faiss')):
    st.write("Loading existing vector store...")
    vectorstore = FAISS.load_local(INDEX_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
    print("Vectorstore successfully loaded from disk.", vectorstore)
else:
    st.write("No existing index found. Please run the ingestion process to create a new vector store.")
    vectorstore = None
    # Update vectorstore only if explicitly requested


# Creating retrieval-based QA chain
if vectorstore is not None:
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())
else:
    qa_chain = None

# User input for questions
user_query = st.text_input("Ask a question about the PDF documents or request a tool:")
if user_query and qa_chain is not None:
    # Retrieve relevant documents and generate an answer using the QA chain
    with st.spinner("Fetching answer..."):
        retrieved_answer = qa_chain.invoke(user_query)
        st.write("Retrieved context:")
        st.write(retrieved_answer)
        # Use the agent to generate an enriched response based on the retrieved context
        try:
            answer = agent.invoke(user_query + ' Context: ' + retrieved_answer['result'])
        except ValueError as e:
            st.write("An error occurred while parsing the output. Retrying...")
            answer = "Unable to generate a response at the moment, please try again."
        st.write("Final Answer:")
        st.write(answer)

st.sidebar.write("Additional settings and utilities coming soon...")
