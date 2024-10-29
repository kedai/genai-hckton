import os
import json

import requests
import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, Tool
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dp_html_parser import HTMLContentParser
import stanza
from typing import Generator

# Constants and initialization
HTML_DIRECTORY = "html"  # Changed from PDF_DIRECTORY
INDEX_DIRECTORY = "index-dir-html"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
METADATA_FILE = "index-dir/metadata.json"

# Initialize Streamlit state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Initialize core components
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
llm = ChatOllama(model="llava")

# Initialize stanza
model_dir = Path.home() / "stanza_resources" / "en"
if not model_dir.exists():
    stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize')

system_prompt = (
    "You are an assistant that answers questions related to PayNet's APIs and documentation. "
    "If you have sufficient context, provide a final, clear, and actionable answer. "
    "If you cannot answer, provide guidance on the next steps without initiating any code generation or incomplete actions."
)
#    "You must always return valid JSON. Do not return any additional text."


class DocumentProcessor:
    def __init__(self, index_directory: str, html_directory: str, embeddings):
        self.index_directory = Path(index_directory)
        self.html_directory = Path(html_directory)
        self.embeddings = embeddings
        self.logger = self._setup_logger()
        self.url = "http://localhost:11434/api/generate"
        self.model = llm
        self.batch_size = 50

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _get_or_create_vectorstore(self) -> FAISS:
        """Get existing vectorstore or create a new one."""
        if os.path.exists(self.index_directory):
            try:
                return FAISS.load_local(str(self.index_directory), self.embeddings, allow_dangerous_deserialization=True)
            except Exception as e:
                self.logger.error(f"Error loading existing index: {str(e)}")

        # Create empty vectorstore if none exists
        empty_docs = [Document(page_content="", metadata={})]
        vectorstore = FAISS.from_documents(empty_docs, self.embeddings)
        vectorstore.save_local(str(self.index_directory))
        return vectorstore

    def _document_generator(self) -> Generator[Document, None, None]:
        """Generate documents from HTML files incrementally."""
        metadata = self.load_metadata()

        for file_name in os.listdir(self.html_directory):
            if not file_name.endswith(".html"):
                continue

            html_path = self.html_directory / file_name
            last_modified = os.path.getmtime(html_path)

            if file_name in metadata and metadata[file_name].get("last_modified") == last_modified:
                continue

            try:
                # Update status for current file
                st.text(f"Processing file: {file_name}")

                with open(html_path, 'r', encoding='utf-8') as file:
                    html_content = file.read()

                documents = self._process_html(html_content, file_name)
                processed_docs = self._process_documents(documents)

                metadata[file_name] = {"last_modified": last_modified}
                self.save_metadata(metadata)

                for doc in processed_docs:
                    yield doc

            except Exception as e:
                st.error(f"Error processing {file_name}: {str(e)}")
                continue

    def _save_batch(self, vectorstore: FAISS, documents: List[Document]):
        """Save a batch of documents to the vectorstore."""
        if not documents:
            return

        try:
            batch_vectorstore = FAISS.from_documents(documents, self.embeddings)
            vectorstore.merge_from(batch_vectorstore)
            vectorstore.save_local(str(self.index_directory))

        except Exception as e:
            self.logger.error(f"Error saving batch: {str(e)}")
            raise

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

    def _ollama_prompt(self, model: str, prompt: str) -> str:
        """
        Send a message to the Ollama API and get a response, handling streamed multi-part responses.

        :param prompt: The message you want to send to the Ollama model.
        :param model: The model to use (default is 'llama2').
        :param url: The API endpoint URL (default is 'http://localhost:11434/api/generate').
        :return: The complete response from the Ollama model.
        """
        # Define the request data with the model and prompt
        data = {
            "model": model,
            "prompt": prompt
        }

        # Define headers for the request
        headers = {
            "Content-Type": "application/json"
        }

        try:
            # Make the POST request to the API
            response = requests.post(self.url, headers=headers, data=json.dumps(data))

            # Check if the request was successful
            if response.status_code == 200:
                # Split the response content by newlines to handle multiple JSON objects
                response_lines = response.content.decode('utf-8').strip().split("\n")

                # Initialize an empty string to store the complete response
                complete_response = ""

                # Parse each line as a separate JSON object
                for line in response_lines:
                    try:
                        json_data = json.loads(line)
                        # Append the 'response' field to the complete response
                        complete_response += json_data.get('response', '')
                    except json.JSONDecodeError:
                        return f"Error decoding JSON: {line}"

                return complete_response
            else:
                return f"Failed to get response. Status code: {response.status_code}, Error: {response.text}"

        except Exception as e:
            return f"An error occurred: {str(e)}"



    def ingest_html(self) -> Optional[FAISS]:
        """Incrementally ingest HTML files and update vectorstore."""
        try:
            # Count total files to process first
            total_files = len([f for f in os.listdir(self.html_directory) if f.endswith('.html')])
            if total_files == 0:
                st.warning("No HTML files found in directory.")
                return None, 0

            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics = st.empty()

            vectorstore = self._get_or_create_vectorstore()
            doc_batch = []
            total_processed = 0
            files_processed = 0

            for doc in self._document_generator():
                doc_batch.append(doc)
                total_processed += 1

                # Update metrics in real-time
                with metrics.container():
                    col1, col2 = st.columns(2)
                    col1.metric("Documents Processed", total_processed)
                    col2.metric("Current Batch Size", len(doc_batch))

                # Save batch when it reaches the specified size
                if len(doc_batch) >= self.batch_size:
                    status_text.text(f"Saving batch of {len(doc_batch)} documents...")
                    self._save_batch(vectorstore, doc_batch)
                    status_text.text(f"Saved batch of {len(doc_batch)} documents. Total processed: {total_processed}")
                    doc_batch = []
                    files_processed += 1
                    # Update progress bar
                    progress_bar.progress(min(files_processed / total_files, 1.0))

            # Save any remaining documents
            if doc_batch:
                status_text.text(f"Saving final batch of {len(doc_batch)} documents...")
                self._save_batch(vectorstore, doc_batch)
                status_text.text(f"Saved final batch of {len(doc_batch)} documents. Total processed: {total_processed}")
                files_processed += 1
                progress_bar.progress(1.0)

            # Clear temporary status displays
            status_text.empty()
            metrics.empty()
            progress_bar.empty()

            return vectorstore, total_processed

        except Exception as e:
            self.logger.error(f"Error during incremental ingestion: {str(e)}")
            return None, 0


    def _process_html(self, html_content: str, file_name: str) -> List[Document]:
        """Process a single HTML document."""
        documents = []
        parser = HTMLContentParser(html_content)
        print(f"file_name: {file_name}")
        # Extract main content (headings and paragraphs)
        contents = parser.get_contents()
        print(f"Contents: {' '.join(contents)}")
        if contents:
            documents.append(Document(
                    page_content='\n'.join(contents),
                    metadata={"semantic_tag": "general_text", "source": file_name}
                ))

        # Extract tables
        tables = parser.get_tables()
        for table in tables:
            table_content = f"Headers: {', '.join(table['headers'])}\nData: {str(table['rows'])}"
            print(f"Table content: {table_content}")
            prompt = f"""
            1. summarize and annotate  the table below.
            2. if the table is a flow, list the steps clearly.
                    the table is from paynet's api reference and documentation: {table_content}"""
            table_meaning = self._ollama_prompt(self.model, prompt)
            print(f"Table meaning: {table_meaning}")
            if table_content.strip():
                documents.append(Document(
                    page_content=table_meaning,
                    metadata={"semantic_tag": "table_data", "source": file_name}
                ))

        # Extract images (alt text)
        images = parser.get_images()
        for image in images:
            print(image)
            if image['alt'].strip():
                documents.append(Document(
                    page_content=f"Image: {image['alt']}",
                    metadata={"semantic_tag": "image_alt_text", "source": file_name}
                ))

        return documents


# Initialize document processor
doc_processor = DocumentProcessor(INDEX_DIRECTORY, HTML_DIRECTORY, embeddings)

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

# HTML Ingestion Button
if st.button("Ingest HTML Files"):
    # Create a container for the ingestion process
    ingestion_container = st.container()

    with ingestion_container:
        st.write("Starting HTML ingestion process...")
        vectorstore, doc_count = doc_processor.ingest_html()

        if vectorstore is not None:
            st.session_state.vectorstore = vectorstore
            # Update QA chain with new vectorstore
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                return_source_documents=True
            )
            st.success(f"Successfully processed {doc_count} documents!")
        else:
            st.error("Failed to process HTML files.")

def generate_code_snippet(query_with_context: str) -> str:
    parts = query_with_context.split('Context:', 1)
    query = parts[0].strip()
    context = parts[1].strip() if len(parts) > 1 else ""

    code_example = f"""# Code example based on the following context:
# {context}

import requests

def handle_paynet_api():
    try:
        response = requests.get('https://api.paynet.example/resource')
        return response.json()
    except Exception as e:
        return None"""

    return code_example

def extract_api_details(query_with_context: str) -> str:
    parts = query_with_context.split('Context:', 1)
    context = parts[1].strip() if len(parts) > 1 else ""

    return f"""Based on the provided information:

{context}

Additional API Usage Recommendations:
1. Always follow the documented best practices
2. Implement proper error handling
3. Monitor API responses for any issues
4. Keep your API implementation up to date"""

def enhance_response(query_with_context: str) -> str:
    parts = query_with_context.split('Context:', 1)
    context = parts[1].strip() if len(parts) > 1 else ""

    return f"""Based on the retrieved information and analysis:

KEY FINDINGS:
{context}

RECOMMENDATIONS:
1. Review and address each identified issue
2. Follow best practices
3. Implement proper error handling
4. Regular testing and monitoring

For complete guidelines, please consult the official documentation."""

# Define tools
tools = [
#    Tool(
#        name="Code Generator",
#        func=generate_code_snippet,
#        description="Generate code snippets related to the API documentation while considering the context."
#    ),
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
user_query = st.text_input("Ask a question about the HTML documents or request a tool:")

if user_query:
    if st.session_state.qa_chain is None:
        st.warning("Please ingest HTML files first by clicking the 'Ingest HTML Files' button above.")
    else:
        with st.spinner("Fetching answer..."):
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

st.write("This app uses an RAG approach to answer questions from HTML files stored in a directory.")
