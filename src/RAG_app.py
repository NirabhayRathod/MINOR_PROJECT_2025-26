import os 
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM
LLM = ChatGroq(model='Gemma2-9b-It', groq_api_key=groq_api_key)

# Prompt template for detailed answers
prompt = ChatPromptTemplate.from_template(
    '''Answer the question based on provided context only.
    Provide a detailed and comprehensive response.
    <context>{context}</context>
    Question: {input}
    '''
)

# RAG initialization function with FAISS caching
def RAG_function():
    if 'vectors' not in st.session_state:
        st.session_state.embedding = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en')
        faiss_index_path = "faiss_index"

        # If saved FAISS index exists, load it safely
        if os.path.exists(faiss_index_path):
            st.session_state.vectors = FAISS.load_local(
                faiss_index_path,
                st.session_state.embedding,
                allow_dangerous_deserialization=True
            )
        else:
            # Load and split the PDF
            st.session_state.loader = PyPDFLoader('indian_laws_final_understandable.pdf')
            st.session_state.doc = st.session_state.loader.load()
            
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,  # slightly bigger chunks
                chunk_overlap=300  # more overlap
            )
            st.session_state.final_doc = st.session_state.text_splitter.split_documents(st.session_state.doc)
            
            # Create FAISS vector store and save it
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_doc, st.session_state.embedding)
            st.session_state.vectors.save_local(faiss_index_path)
       
# Streamlit UI
user_query = st.text_input('Enter your query from document')

if st.button("Document Embedding"):
    RAG_function()
    st.write('✅ Vector Database is ready')
     
if user_query:
    # Use retriever with more chunks for longer answers
    retriever = st.session_state.vectors.as_retriever(search_kwargs={'k':10})
    doc_chain = create_stuff_documents_chain(LLM, prompt)
    retriever_chain = create_retrieval_chain(retriever, doc_chain)
     
    response = retriever_chain.invoke({'input': user_query})
    
    st.write("**Answer:**")
    st.write(response['answer'])
    
    # Optional: Show retrieved context
    with st.expander('Document similarity search'):
        for i, doc in enumerate(response['context']):
            st.write(f"**Chunk {i+1}:**")
            st.write(doc.page_content)
            st.write('--------------------------------')
