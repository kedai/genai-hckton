#!/usr/bin/env python3
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from document_processor import DocumentProcessor
from config import (
    HTML_DIRECTORY,
    INDEX_DIRECTORY,
    embeddings,
    llm,
    VECTOR_SEARCH_INDEX_NAME,
    VECTOR_SEARCH_ENDPOINT_NAME,
    vector_store
)

# Conversation prompt template
CONVERSATION_PROMPT = PromptTemplate(
    template="""You are a helpful assistant for PayNet's API documentation. Use the following context and chat history to provide accurate, relevant answers.
    If you cannot answer based on the context, say so clearly.

    Context: {context}

    Chat History: {chat_history}

    Question: {question}

    Helpful Answer:""",
    input_variables=["context", "chat_history", "question"]
)

def initialize_conversation_chain(vectorstore):
    """Initialize the conversation chain with memory."""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": CONVERSATION_PROMPT},
        verbose=True
    )

# Initialize Streamlit state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'doc_processor' not in st.session_state:
    st.session_state.doc_processor = DocumentProcessor(
        html_directory=HTML_DIRECTORY,
        index_directory=INDEX_DIRECTORY,
        embeddings=embeddings,
        llm=llm
    )

def main():
    st.title("PayNet API Documentation Assistant")

    # Sidebar for ingestion
    with st.sidebar:
        if st.button("Reprocess HTML Files"):
            ingestion_container = st.container()
            with ingestion_container:
                st.write("Starting HTML ingestion process...")
                vectorstore, doc_count = st.session_state.doc_processor.ingest_html()

                if vectorstore is not None:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.conversation_chain = initialize_conversation_chain(vectorstore)
                    st.success(f"Successfully processed {doc_count} documents!")
                else:
                    st.error("Failed to process HTML files.")

        st.markdown("---")
        st.markdown("""
        ### About this Assistant
        This documentation assistant uses RAG (Retrieval-Augmented Generation) to:
        - Search through documentation
        - Maintain conversation context
        - Provide source references
        - Give accurate, contextual answers
        """)

    # Try to load existing vectorstore if not already loaded
    if st.session_state.vectorstore is None:
        try:
            vectorstore = st.session_state.doc_processor.get_or_create_vectorstore()
            if vectorstore is not None:
                st.session_state.vectorstore = vectorstore
                st.session_state.conversation_chain = initialize_conversation_chain(vectorstore)
        except Exception as e:
            st.error(f"Error initializing vectorstore: {str(e)}")

    # Main chat interface
    user_question = st.text_input("Ask a question about the PayNet API:", key="user_question")

    if user_question:
        if st.session_state.conversation_chain is None:
            st.warning("Please ingest HTML files first using the button in the sidebar.")
        else:
            try:
                with st.spinner("Searching documentation..."):
                    response = st.session_state.conversation_chain({"question": user_question})

                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(response["answer"])

                    # Display sources
                    if response.get("source_documents"):
                        with st.expander("View Sources"):
                            for idx, doc in enumerate(response["source_documents"], 1):
                                st.markdown(f"**Source {idx}:**")
                                st.markdown(doc.page_content)
                                st.markdown(f"*From: {doc.metadata.get('source', 'Unknown')} "
                                          f"(Type: {doc.metadata.get('type', 'Unknown')})*")
                                st.markdown("---")

                    # Update chat history
                    st.session_state.chat_history.append((user_question, response["answer"]))

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.warning("Please try rephrasing your question.")

    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### Recent Conversations")
        for question, answer in reversed(st.session_state.chat_history[-5:]):
            with st.expander(f"Q: {question[:50]}...", expanded=False):
                st.markdown("**Question:**")
                st.markdown(question)
                st.markdown("**Answer:**")
                st.markdown(answer)

if __name__ == "__main__":
    main()