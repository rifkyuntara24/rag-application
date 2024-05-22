import streamlit as st
from models.model import initialize_llm, initialize_embeddings, initialize_vectorstore, create_rag_chain
from langchain_community.document_loaders import PyPDFLoader
import tempfile

def set_api_key():
    groq_api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")
    if groq_api_key:
        st.session_state["GROQ_API_KEY"] = groq_api_key  # Store the API key in the session state

def initialize_rag_model():
    if "GROQ_API_KEY" in st.session_state:
        try:
            llm = initialize_llm(st.session_state["GROQ_API_KEY"])
            embeddings = initialize_embeddings()
            st.session_state["llm"] = llm
            st.session_state["embeddings"] = embeddings
            st.success("API key set successfully.")
        except Exception as e:
            st.error(f"Failed to initialize model: {e}")

def upload_and_process_documents():
    uploaded_files = st.sidebar.file_uploader("Upload your PDF documents", accept_multiple_files=True, type=['pdf'])
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            bytes_data = uploaded_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(bytes_data)
                pdf_loader = PyPDFLoader(temp_file.name)
                documents.extend(pdf_loader.load())
        if documents:
            try:
                if "retriever" not in st.session_state:
                    retriever = initialize_vectorstore(documents, st.session_state["embeddings"])
                    st.session_state["retriever"] = retriever
                    st.session_state["documents_loaded"] = True
                    st.write("Documents uploaded and processed successfully.")
                else:
                    st.write("Documents have already been uploaded and processed.")
            except Exception as e:
                st.error(f"Failed to process documents: {e}")
        else:
            st.write("No valid documents found in the uploaded files.")
    else:
        if "documents_loaded" not in st.session_state:
            st.write("No documents available for retrieval. Please upload documents.")

def handle_query():
    input_text = st.text_area("Enter your query:")
    if st.button("Get Answer"):
        if input_text:
            try:
                if "retriever" in st.session_state and "llm" in st.session_state:
                    rag_chain = create_rag_chain(st.session_state["retriever"], st.session_state["llm"])
                    response = rag_chain.invoke({"input": input_text})
                    st.write("### Answer")
                    st.write(response['answer'])
                else:
                    st.error("Model or retriever is not initialized.")
            except Exception as e:
                st.error(f"Failed to retrieve answer: {e}")
        else:
            st.write("Please enter a query.")

def main():
    st.title("Tanya Aku Aja!🚀")
    st.write("Rifkuy yang bikin aku ada")
    st.markdown("[Tentang Rifkuy di sini!](https://www.linkedin.com/in/muhammad-rifky-untara-858ab3228/)")
    
    st.sidebar.title("Settings")
    st.sidebar.markdown("[Get Groq API HERE!](https://console.groq.com/keys)")
    set_api_key()
    
    if "GROQ_API_KEY" in st.session_state:
        initialize_rag_model()
        st.sidebar.title("Document Upload")
        upload_and_process_documents()
        handle_query()
    else:
        st.sidebar.warning("Please enter your Groq API key.")

if __name__ == "__main__":
    main()
