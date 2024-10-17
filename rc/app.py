import os
import io
import json
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, Tool
import fitz
import camelot
import pytesseract
from PIL import Image
from transformers import pipeline
import stanza
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Constants and initialization
PDF_DIRECTORY = "data-pdfs"
INDEX_DIRECTORY = "index-dir"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METADATA_FILE = "index-dir/metadata.json"

# Initialize Streamlit state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Initialize core components
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
llm = ChatOllama(model="llama3.2")

# Initialize stanza
model_dir = Path.home() / "stanza_resources" / "en"
if not model_dir.exists():
    stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize')

system_prompt = (
    "You are an assistant that answers questions related to PayNet's APIs. "
    "If you have sufficient context, provide a final, clear, and actionable answer. "
    "If you cannot answer, provide guidance on the next steps without initiating any code generation or incomplete actions."
)

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

class DocumentProcessor:
    def __init__(self, index_directory: str, pdf_directory: str, embeddings):
        self.index_directory = Path(index_directory)
        self.pdf_directory = Path(pdf_directory)
        self.embeddings = embeddings
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def process_image(self, image: Image.Image) -> Optional[str]:
        """Process a single image with OCR and summarization if needed."""
        try:
            ocr_text = pytesseract.image_to_string(image)
            if not ocr_text.strip():
                return None

            if len(ocr_text) > 1024:
                return self._summarize_long_text(ocr_text)
            return ocr_text
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return None

    def _summarize_long_text(self, text: str) -> str:
        """Summarize long text by breaking it into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        summaries = []

        for chunk in chunks:
            try:
                summary = self.summarizer(chunk, max_length=50, min_length=25, do_sample=False)[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                self.logger.error(f"Error summarizing chunk: {str(e)}")
                continue

        return ' '.join(summaries)

    def process_tables(self, pdf_path: str, page_num: int) -> List[Document]:
        """Process tables from a PDF page."""
        documents = []
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=str(page_num + 1))
            for table in tables:
                table_text = table.df.to_string()
                table_summary = self._summarize_long_text(table_text) if len(table_text) > 1024 else table_text
                documents.append(Document(
                    page_content=table_summary,
                    metadata={"semantic_tag": "numerical_data"}
                ))
        except Exception as e:
            self.logger.error(f"Error processing tables on page {page_num}: {str(e)}")
        return documents

    def ingest_pdfs(self) -> Tuple[Optional[FAISS], List[Document]]:
        """Main function to ingest PDFs and create vectorstore."""
        st.write("Starting PDF ingestion...")
        metadata = self.load_metadata()
        documents = []

        for file_name in os.listdir(self.pdf_directory):
            if not file_name.endswith(".pdf"):
                continue

            pdf_path = self.pdf_directory / file_name
            last_modified = os.path.getmtime(pdf_path)

            # Skip if already processed and unchanged
            if file_name in metadata and metadata[file_name].get("last_modified") == last_modified:
                continue

            try:
                doc = fitz.open(pdf_path)
                documents.extend(self._process_pdf(doc, file_name, pdf_path))
                metadata[file_name] = {"last_modified": last_modified}
            except Exception as e:
                self.logger.error(f"Error processing {file_name}: {str(e)}")
                continue

        # Process all documents into sentence-level chunks
        processed_docs = self._process_documents(documents)

        if not processed_docs:
            st.write("No valid documents found in the PDFs for embedding.")
            return None, []

        # Create and save vectorstore
        try:
            vectorstore = FAISS.from_documents(processed_docs, self.embeddings)
            vectorstore.save_local(str(self.index_directory))
            st.write(f"Successfully ingested {len(processed_docs)} documents.")
            self.save_metadata(metadata)
            return vectorstore, processed_docs
        except Exception as e:
            self.logger.error(f"Error creating vectorstore: {str(e)}")
            return None, processed_docs

    def _process_pdf(self, doc, file_name: str, pdf_path: Path) -> List[Document]:
        """Process a single PDF document."""
        documents = []
        for page_num in range(len(doc)):
            st.write(f"Extracting from page {page_num + 1} of {file_name}...")
            page = doc.load_page(page_num)

            # Extract text
            text = page.get_text()
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"semantic_tag": "general_text"}
                ))

            # Process tables
            documents.extend(self.process_tables(pdf_path, page_num))

            # Process images
            documents.extend(self._process_page_images(page, doc))

        return documents

    def _process_page_images(self, page, doc) -> List[Document]:
        """Process images from a PDF page."""
        documents = []
        for img in page.get_images(full=True):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))

                ocr_text = self.process_image(image)
                if ocr_text:
                    documents.append(Document(
                        page_content=ocr_text,
                        metadata={"semantic_tag": "image_extracted_text"}
                    ))
            except Exception as e:
                self.logger.error(f"Error processing image: {str(e)}")
                continue
        return documents

    def _process_documents(self, documents: List[Document]) -> List[Document]:
        """Process documents into sentence-level chunks."""
        processed_docs = []
        for doc in documents:
            if not doc.page_content.strip():
                continue

            try:
                nlp_doc = nlp(doc.page_content)
                for sentence in nlp_doc.sentences:
                    if sentence.text.strip():
                        processed_docs.append(Document(
                            page_content=sentence.text.strip(),
                            metadata={"semantic_tag": doc.metadata["semantic_tag"]}
                        ))
            except Exception as e:
                self.logger.error(f"Error processing document text: {str(e)}")
                continue

        return processed_docs

    def load_metadata(self) -> Dict:
        """Load metadata from file."""
        metadata_file = self.index_directory / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def save_metadata(self, metadata: Dict):
        """Save metadata to file."""
        metadata_file = self.index_directory / "metadata.json"
        self.index_directory.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)



# Initialize document processor

doc_processor = DocumentProcessor(INDEX_DIRECTORY, PDF_DIRECTORY, embeddings)

# Try to load existing vectorstore
if os.path.exists(INDEX_DIRECTORY):
    try:
        st.session_state.vectorstore = FAISS.load_local(INDEX_DIRECTORY,
                                                        embeddings,
                                                        allow_dangerous_deserialization=True)
        # Initialize QA chain with loaded vectorstore
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vectorstore.as_retriever(),
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error loading existing index: {str(e)}")

# PDF Ingestion Button
if st.button("Ingest PDFs"):
    with st.spinner("Processing PDFs..."):
        vectorstore, documents = doc_processor.ingest_pdfs()
        if vectorstore is not None:
            st.session_state.vectorstore = vectorstore
            # Update QA chain with new vectorstore
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            st.success(f"Successfully processed {len(documents)} documents!")
        else:
            st.error("Failed to process PDFs.")


# Define custom tools
# Enhanced tool definitions with context preservation
def generate_code_snippet(query_with_context: str) -> str:
    """Generate code snippets while preserving context from retrieved documents."""
    # Split the input to separate query from context
    parts = query_with_context.split('Context:', 1)
    query = parts[0].strip()
    context = parts[1].strip() if len(parts) > 1 else ""

    # Generate code example based on both query and context
    code_example = f"""# Code example based on the following context:
# {context}

import requests

def handle_paynet_api():
    # Implement best practices based on known issues:
    # 1. Avoid load testing in sandbox
    # 2. Check backward compatibility when updating

    try:
        response = requests.get('https://api.paynet.example/resource')
        return response.json()
    except Exception as e:
        return None"""

    return code_example

def extract_api_details(query_with_context: str) -> str:
    """Extract and enhance API details while preserving context."""
    # Split the input to separate query from context
    parts = query_with_context.split('Context:', 1)
    query = parts[0].strip()
    context = parts[1].strip() if len(parts) > 1 else ""

    # Format and enhance the context
    enhanced_details = f"""Based on the provided information:

{context}

Additional API Usage Recommendations:
1. Always follow the documented best practices
2. Implement proper error handling
3. Monitor API responses for any issues
4. Keep your API implementation up to date"""

    return enhanced_details

def enhance_response(query_with_context: str) -> str:
    """Enhance the response by preserving and structuring the context."""
    # Split the input to separate query from context
    parts = query_with_context.split('Context:', 1)
    query = parts[0].strip()
    context = parts[1].strip() if len(parts) > 1 else ""

    # Structure and enhance the response
    enhanced_response = f"""Based on the retrieved information and analysis:

KEY FINDINGS:
{context}

RECOMMENDATIONS:
1. Review and address each identified issue:
   - Carefully consider the points mentioned in the context
   - Implement appropriate solutions for each issue

2. Best Practices to Follow:
   - Follow official documentation guidelines
   - Implement proper error handling
   - Regular testing and monitoring
   - Keep systems up to date

ADDITIONAL CONSIDERATIONS:
- Document any changes or implementations
- Monitor system performance
- Maintain security best practices
- Regular review and updates

For complete guidelines and best practices, please consult the official documentation."""

    return enhanced_response

# Define enhanced tools
tools = [
    Tool(
        name="Code Generator",
        func=generate_code_snippet,
        description="Generate code snippets related to the API documentation while considering the context."
    ),
    Tool(
        name="API Detail Extractor",
        func=extract_api_details,
        description="Extract and enhance detailed API information such as parameters, endpoints, etc."
    ),
    Tool(
        name="Response Enhancer",
        func=enhance_response,
        description="Enhance and structure the response with detailed context and recommendations."
    )
]

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate"
)

# Question-answering interface
user_query = st.text_input("Ask a question about the PDF documents or request a tool:")

if user_query:
    if st.session_state.qa_chain is None:
        st.warning("Please ingest PDFs first by clicking the 'Ingest PDFs' button above.")
    else:
        with st.spinner("Fetching answer..."):
            # try:
                # Get answer from QA chain
            qa_response = st.session_state.qa_chain({"query": user_query})

            # Display source documents
            st.subheader("Retrieved Context:")
            for doc in qa_response.get("source_documents", []):
                st.write(doc.page_content)
                st.write("---")

            # Use the enhanced agent to generate a comprehensive response
            enhanced_prompt = f"{system_prompt}\n\n {user_query} Context: {qa_response['result']}"
            enhanced_response = agent.invoke({
                "input": enhanced_prompt,
                "tool": "Response Enhancer"
            })

            st.subheader("Enhanced Answer:")
            st.write(enhanced_response)
            #st.write(enhanced_response.get('output', enhanced_response))

            # except Exception as e:
            #     st.error(f"An error occurred: {str(e)}")
            #     st.write("Falling back to simple response...")
            #     st.write(qa_response.get('result', "No answer available"))

st.write("This app uses an RAG approach to answer questions from PDFs stored in a directory.")
