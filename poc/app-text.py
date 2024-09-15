import logging
import os
import glob

import streamlit as st
from charset_normalizer import from_path
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="AI-Powered Assistant", layout="wide")

# Sidebar for user configuration
st.sidebar.title("Settings")
storage_input = st.sidebar.text_input("Storage Directory", "./faiss_storage")
required_exts = st.sidebar.multiselect("File Extensions", [".txt", ".md"], default=[".txt"])
file_directory = st.sidebar.text_input("Directory", "./txts")
# Title of the app
st.title("AI-Powered Assistant")

# -----------------------------
# Custom Text Loader Definition
# -----------------------------
class CustomTextLoader(TextLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path)

    def load(self):
        try:
            # Use charset-normalizer to detect encoding and read the file
            result = from_path(self.file_path).best()
            if result:
                text = result.encoding  # Corrected method to retrieve text
                logger.info(f"Successfully loaded file: {self.file_path} with encoding: {result.encoding}")
                # Return a list with a single Document object
                return [Document(page_content=text, metadata={"source": self.file_path})]
            else:
                logger.warning(f"Unable to detect encoding for file: {self.file_path}. Attempting fallback encoding.")
                # Fallback to reading with 'latin-1' encoding if detection fails
                with open(self.file_path, 'r', encoding='latin-1') as f:
                    text = f.read()
                logger.info(f"Successfully loaded file with fallback encoding: {self.file_path}")
                return [Document(page_content=text, metadata={"source": self.file_path})]
        except Exception as e:
            # Log the error and skip the problematic file
            logger.error(f"Error loading file {self.file_path}: {e}")
            return []  # Return an empty list to indicate failure for this file

# -----------------------------
# Resource Loading Function
# -----------------------------
@st.cache_resource
def load_resources(storage: str, extensions: list):
    """
    Load documents from the specified directory, handle different encodings,
    and create or load a FAISS vector store.

    Args:
        storage (str): Path to store/load the FAISS vector store.
        extensions (list): List of file extensions to include.

    Returns:
        ConversationalRetrievalChain or None: The QA chain if successful, else None.
    """
    # Initialize HuggingFace embeddings model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/distilbert-base-nli-mean-tokens")

    if not os.path.exists(storage):
        # Define the directory containing your text files
        file_path = "./rfcs"  # You can make this configurable if needed
        files = []
        for ext in extensions:
            pattern = os.path.join(file_path, "**", f"*{ext}")
            matched_files = glob.glob(pattern, recursive=True)
            files.extend(matched_files)

        logger.info(f"Found {len(files)} files with extensions {extensions} in {file_path}")

        documents = []
        failed_files = []

        for file in files:
            loader = CustomTextLoader(file)
            loaded_docs = loader.load()
            if loaded_docs:
                documents.extend(loaded_docs)
            else:
                failed_files.append(file)

        if failed_files:
            st.warning(f"Failed to load {len(failed_files)} files. Check logs for details.")
            logger.warning(f"Failed to load files: {failed_files}")

        if not documents:
            st.error("No documents found or all documents failed to load!")
            return None

        # Create FAISS vector store with the embeddings
        vectorstore = FAISS.from_documents(documents, embeddings_model)
        vectorstore.save_local(storage)
        logger.info(f"Vector store created and saved to {storage}")
    else:
        # Load the vector store from storage
        try:
            vectorstore = FAISS.load_local(
                storage,
                HuggingFaceEmbeddings(model_name="sentence-transformers/distilbert-base-nli-mean-tokens"),
                allow_dangerous_deserialization=True
            )
            logger.info(f"Vector store loaded from {storage}")
        except Exception as e:
            st.error(f"Error loading vector store from {storage}: {e}")
            logger.error(f"Error loading vector store from {storage}: {e}")
            return None

    # Initialize Ollama LLM
    try:
        llm = ChatOllama(
            model="llama3.1",
            temperature=0,
        )
        logger.info("Ollama LLM initialized successfully.")
    except Exception as e:
        st.error(f"Error initializing Ollama LLM: {e}")
        logger.error(f"Error initializing Ollama LLM: {e}")
        return None

    # Set up memory to keep track of conversation history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # System prompt for guiding the model's behavior
    system_message = SystemMessagePromptTemplate.from_template(
        """You are a RFC assistant. You provide concise,
        informative, and accurate responses to request for comments queries."""
    )

    # Create a Conversational QA chain using the Ollama model and memory
    try:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        logger.info("Conversational Retrieval Chain created successfully.")
    except Exception as e:
        st.error(f"Error creating Conversational Retrieval Chain: {e}")
        logger.error(f"Error creating Conversational Retrieval Chain: {e}")
        return None

    return qa_chain

# -----------------------------
# Main Application Function
# -----------------------------
def main():
    # Load the QA chain (Vectorstore + Ollama LLM)
    qa_chain = load_resources(storage=storage_input, extensions=required_exts)

    if qa_chain is None:
        st.stop()  # Stop further execution if resources failed to load

    # Initialize or retrieve chat history from session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Input field for user queries
    query = st.text_input("Ask a question:", "")

    if query:
        with st.spinner("Generating response..."):
            try:
                # Generate a response using the QA chain
                response = qa_chain({"question": query, "chat_history": st.session_state['chat_history']})

                # Display the response
                st.write(f"**Response:** {response['answer']}")

                # Update the chat history
                st.session_state['chat_history'] = response['chat_history']
            except Exception as e:
                st.error(f"Error generating response: {e}")
                logger.error(f"Error generating response for query '{query}': {e}")

# -----------------------------
# Run the Streamlit App
# -----------------------------
if __name__ == "__main__":
    main()
