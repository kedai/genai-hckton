#!/usr/bin/env python3
import streamlit as st
import time
from datetime import datetime, timedelta
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from document_processor import DocumentProcessor
import streamlit.components.v1 as components
from config import (
    HTML_DIRECTORY,
    INDEX_DIRECTORY,
    embeddings,
    VECTOR_SEARCH_INDEX_NAME,
    VECTOR_SEARCH_ENDPOINT_NAME,
    vector_store,
    create_llm
)

# Enhanced conversation prompt template with better context handling
CONVERSATION_PROMPT = PromptTemplate(
    template="""You are a helpful assistant for PayNet's API documentation. Use the following context and chat history to provide accurate, relevant answers.
    If you cannot answer based on the context, say so clearly. Always cite the specific parts of the documentation you're referencing. Rephrase the question if you must to get the best answer.

    Context: {context}

    Chat History: {chat_history}

    Question: {question}

    Helpful Answer:""",
    input_variables=["context", "chat_history", "question"]
)

def initialize_conversation_chain(vectorstore):
    """Initialize the conversation chain with memory and enhanced retrieval."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    # Create new LLM instance
    llm = create_llm()

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={
            "k": 4,
            "search_type": "similarity",
            "score_threshold": 0.5  # Only return relevant results
        }),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": CONVERSATION_PROMPT},
        verbose=True
    )


def format_time_elapsed(start_time):
    """Format the elapsed time in a human-readable format."""
    if not start_time:
        return ""
    elapsed = datetime.now() - start_time
    if elapsed < timedelta(minutes=1):
        return f"{elapsed.seconds} seconds"
    elif elapsed < timedelta(hours=1):
        return f"{elapsed.seconds // 60} minutes {elapsed.seconds % 60} seconds"
    else:
        hours = elapsed.seconds // 3600
        minutes = (elapsed.seconds % 3600) // 60
        return f"{hours} hours {minutes} minutes"

def display_processing_status():
    """Display current processing status with better formatting."""
    status = st.session_state.get('processing_status', {})
    if not status:
        return

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Files Processed",
            f"{status.get('processed_files', 0)}/{status.get('total_files', 0)}"
        )
        if status.get('start_time'):
            st.text(f"Time Elapsed: {format_time_elapsed(status['start_time'])}")

    with col2:
        st.metric("Documents Processed", status.get('total_documents', 0))
        if status.get('current_batch', 0) > 0:
            st.text(f"Current Batch: {status['current_batch']}")

    if status.get('errors', []):
        with st.expander("Processing Errors", expanded=False):
            for error in status['errors']:
                st.error(error)

def initialize_app_state():
    """Initialize application state with better error handling and status tracking."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.vectorstore = None
        st.session_state.conversation_chain = None
        st.session_state.chat_history = []
        st.session_state.processing_status = {
            'current_file': '',
            'total_files': 0,
            'processed_files': 0,
            'total_documents': 0,
            'current_batch': 0,
            'start_time': None,
            'errors': []
        }

    if not st.session_state.initialized:
        try:
            # Initialize document processor
            doc_processor = DocumentProcessor(
                html_directory=HTML_DIRECTORY,
                index_directory=INDEX_DIRECTORY,
                embeddings=embeddings,
                llm=create_llm()  # Create new LLM instance for document processor
            )

            # Attempt to get existing or create new vector store
            with st.spinner("Initializing documentation assistant..."):
                vector_store = doc_processor.get_or_create_vectorstore()

                if vector_store:
                    st.session_state.vectorstore = vector_store
                    st.session_state.doc_processor = doc_processor
                    st.session_state.conversation_chain = initialize_conversation_chain(vector_store)
                    st.session_state.initialized = True
                    st.success("Documentation assistant initialized successfully!")
                else:
                    st.error("Failed to initialize vector store. Please check the logs.")

        except Exception as e:
            st.error(f"Error initializing application: {str(e)}")
            st.info("Try using the 'Reprocess HTML Files' button to reinitialize the system.")

def add_enter_key_handler():
    components.html(
        """
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.key == 'Enter' && !e.shiftKey) {
                e.preventDefault();
                document.querySelector('button[kind="primary"]').click();
            }
        });
        </script>
        """,
        height=0,
    )

def display_chat_interface():
    """Display the chat interface with textarea and improved button position."""
    # Create a container for better layout control
    container = st.container()

    with container:
        user_question = st.text_area(
            "Ask a question about the PayNet API:",
            key="user_question",
            height=100,
            label_visibility="visible"
        )

        # Right-align the button using columns
        _, _, col3 = st.columns([4, 1, 1])
        with col3:
            submit = st.button("Submit", type="primary", key="submit_button")

    add_enter_key_handler()

    if submit and user_question:
        if not st.session_state.conversation_chain:
            st.warning("Please wait for the documentation assistant to initialize or try reprocessing the files.")
            return

        try:
            with st.spinner("Searching documentation..."):
                response = st.session_state.conversation_chain.invoke({"question": user_question})

                st.markdown("### Answer")
                st.markdown(response["answer"])

                if response.get("source_documents"):
                    with st.expander("View Sources", expanded=False):
                        for idx, doc in enumerate(response["source_documents"], 1):
                            st.markdown(f"**Source {idx}:**")
                            st.markdown(doc.page_content)
                            st.markdown(f"*From: {doc.metadata.get('source', 'Unknown')} "
                                      f"(Type: {doc.metadata.get('type', 'Unknown')})*")
                            st.markdown("---")

                st.session_state.chat_history.append((user_question, response["answer"]))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.warning("Please try rephrasing your question or reprocessing the documentation if the error persists.")

def display_chat_history():
    """Display chat history with better formatting and organization."""
    if st.session_state.chat_history:
        st.markdown("### Recent Conversations")
        for idx, (question, answer) in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"Q: {question[:50]}{'...' if len(question) > 50 else ''}", expanded=False):
                st.markdown("**Question:**")
                st.markdown(question)
                st.markdown("**Answer:**")
                st.markdown(answer)

def main():
    st.title("PayNet API Documentation Assistant")

    # Sidebar for controls and information
    with st.sidebar:
        if st.button("Reprocess HTML Files"):
            st.session_state.initialized = False
            st.session_state.vectorstore = None
            st.session_state.conversation_chain = None
            st.session_state.processing_status = {
                'current_file': '',
                'total_files': 0,
                'processed_files': 0,
                'total_documents': 0,
                'current_batch': 0,
                'start_time': datetime.now(),
                'errors': []
            }

            # Re-initialize with progress tracking
            initialize_app_state()

        st.markdown("---")
        st.markdown("""
        ### About this Assistant
        This documentation assistant uses RAG (Retrieval-Augmented Generation) to:
        - Search through documentation
        - Maintain conversation context
        - Provide source references
        - Give accurate, contextual answers

        ### Usage Tips
        - Be specific in your questions
        - Reference specific API features or endpoints
        - Check the source references for details
        - Use the 'Reprocess' button if answers seem outdated
        """)

        # Display processing status in sidebar
        if st.session_state.get('processing_status', {}).get('start_time'):
            st.markdown("---")
            st.markdown("### Processing Status")
            display_processing_status()

    # Initialize application state
    initialize_app_state()

    # Main chat interface
    if st.session_state.initialized:
        display_chat_interface()
        display_chat_history()
    else:
        st.info("Please wait while the documentation assistant initializes...")

if __name__ == "__main__":
    main()