# import os
# import streamlit as st
# import pickle
# import time
# from langchain.chat_models import ChatOpenAI  # Updated to use ChatOpenAI
# from langchain.chains import RetrievalQAWithSourcesChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import UnstructuredURLLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# import faiss
# from dotenv import load_dotenv
# import requests
# from bs4 import BeautifulSoup
# from langchain.schema import Document  # Import the Document class
#
# # Load environment variables from env.txt (env.txt should be in the same folder as main.py)
# load_dotenv("env.txt")
#
# # Streamlit setup for user input
# st.title("News Research Tool 📈")
# st.sidebar.title("News Article URLs")
#
# # Input URLs from the user via the sidebar
# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i + 1}")
#     if url:  # Only append non-empty URLs
#         urls.append(url)
#
# process_url_clicked = st.sidebar.button("Process URLs")
# index_file_path = "../vector_index.faiss"
# metadata_file_path = "../vector_index_meta.pkl"
#
# main_placeholder = st.empty()
#
# # Load the OpenAI API key from environment variables
# api_key = os.getenv("OPENAI_API_KEY")
# if not api_key:
#     st.error("API key not found. Please set the OPENAI_API_KEY in env.txt.")
#     st.stop()
#
# # Initialize the ChatOpenAI model
# llm = ChatOpenAI(openai_api_key=api_key, temperature=0.9, max_tokens=500, model_name="gpt-3.5-turbo")
#
# # Alternative data loading method (fallback)
# def alternative_data_loading(url):
#     """Fallback method to load content using requests and BeautifulSoup."""
#     try:
#         response = requests.get(url)
#         if response.status_code == 200:
#             soup = BeautifulSoup(response.content, "html.parser")
#             text = soup.get_text()
#             return Document(page_content=text, metadata={"source": url})
#         else:
#             return None
#     except Exception as e:
#         print(f"Error loading URL {url}: {e}")
#         return None
#
# # If the user clicks "Process URLs"
# if process_url_clicked:
#     # Try loading data using UnstructuredURLLoader
#     try:
#         loader = UnstructuredURLLoader(urls=urls)
#         data = loader.load()
#     except Exception as e:
#         print(f"UnstructuredURLLoader failed with error: {e}")
#         data = []
#
#     # If UnstructuredURLLoader fails, fallback to the alternative method
#     if not data or all(len(d.page_content.strip()) == 0 for d in data):
#         st.warning("UnstructuredURLLoader failed to load data. Trying alternative method.")
#         data = []
#         for url in urls:
#             document = alternative_data_loading(url)
#             if document:
#                 data.append(document)
#             else:
#                 st.error(f"Failed to load content from {url}. Please check the URL.")
#                 st.stop()
#
#     # Split data into smaller chunks using RecursiveCharacterTextSplitter
#     text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
#     main_placeholder.text("Splitting text into smaller chunks...✅")
#     docs = text_splitter.split_documents(data)
#
#     if not docs:
#         st.error("Failed to split documents. Please check the data content.")
#         st.stop()
#
#     # Create embeddings using OpenAI's Embeddings API
#     embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#     embeddings_list = embeddings.embed_documents([doc.page_content for doc in docs])
#
#     if not embeddings_list:
#         st.error("Failed to generate embeddings. Please check your API key and document content.")
#         st.stop()
#
#     # Store embeddings in FAISS vectorstore
#     vectorstore_openai = FAISS.from_documents(docs, embeddings)
#     main_placeholder.text("Building the FAISS index...✅")
#     time.sleep(2)
#
#     # Save the FAISS index and metadata
#     faiss.write_index(vectorstore_openai.index, index_file_path)
#     metadata = {
#         "docstore": vectorstore_openai.docstore,
#         "index_to_docstore_id": vectorstore_openai.index_to_docstore_id
#     }
#     with open(metadata_file_path, "wb") as f:
#         pickle.dump(metadata, f)
#
# # Input for user query
# query = main_placeholder.text_input("Enter your query: ")
# if query:
#     if os.path.exists(index_file_path) and os.path.exists(metadata_file_path):
#         # Load the FAISS index from saved files
#         loaded_index = faiss.read_index(index_file_path)
#
#         # Load the metadata
#         with open(metadata_file_path, "rb") as f:
#             metadata = pickle.load(f)
#
#         # Recreate the FAISS vectorstore
#         embeddings = OpenAIEmbeddings(openai_api_key=api_key)
#         vectorstore = FAISS(
#             index=loaded_index,
#             embedding_function=embeddings.embed_query,
#             docstore=metadata["docstore"],
#             index_to_docstore_id=metadata["index_to_docstore_id"]
#         )
#
#         # Create the retrieval chain for answering questions
#         chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
#         result = chain({"question": query}, return_only_outputs=True)
#
#         # Display the answer and sources
#         st.header("Answer")
#         st.write(result["answer"])
#
#         sources = result.get("sources", "")
#         if sources:
#             st.subheader("Sources:")
#             for source in sources.split("\n"):
#                 st.write(source)

import os
import streamlit as st
import pickle
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import faiss
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document

# Custom CSS for aesthetics
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stSidebar {
        background-color: #f4f7fc;
    }
    .title-text {
        color: #2a3e80;
        font-family: 'Helvetica', sans-serif;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subheader-text {
        color: #4a5568;
        font-family: 'Arial', sans-serif;
        font-size: 1.25rem;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #2a3e80 !important;
        color: white !important;
        border-radius: 5px;
        padding: 5px 15px;
        font-size: 1rem;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        border: 2px solid #2a3e80 !important;
        border-radius: 8px;
        padding: 5px;
    }
    .stTextInput>div>label {
        color: #4a5568;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load environment variables from env.txt (env.txt should be in the same folder as main.py)
load_dotenv("env.txt")

# Streamlit setup for user input
st.markdown("<h1 class='title-text'>News Research Tool 📈</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader-text'>Extract insights and find relevant answers from news articles.</p>", unsafe_allow_html=True)

# Sidebar title
st.sidebar.title("📰 News Article URLs")

# Input URLs from the user via the sidebar
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}", placeholder="Enter a news article URL here")
    if url:  # Only append non-empty URLs
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
index_file_path = "../vector_index.faiss"
metadata_file_path = "../vector_index_meta.pkl"

# Placeholder for status updates
main_placeholder = st.empty()

# Load the OpenAI API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("API key not found. Please set the OPENAI_API_KEY in env.txt.")
    st.stop()

# Initialize the ChatOpenAI model
llm = ChatOpenAI(openai_api_key=api_key, temperature=0.9, max_tokens=500, model_name="gpt-3.5-turbo")

# Alternative data loading method (fallback)
def alternative_data_loading(url):
    """Fallback method to load content using requests and BeautifulSoup."""
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            return Document(page_content=text, metadata={"source": url})
        else:
            return None
    except Exception as e:
        print(f"Error loading URL {url}: {e}")
        return None

# If the user clicks "Process URLs"
if process_url_clicked:
    # Try loading data using UnstructuredURLLoader
    with st.spinner("Processing URLs..."):
        try:
            loader = UnstructuredURLLoader(urls=urls)
            data = loader.load()
        except Exception as e:
            print(f"UnstructuredURLLoader failed with error: {e}")
            data = []

    # If UnstructuredURLLoader fails, fallback to the alternative method
    if not data or all(len(d.page_content.strip()) == 0 for d in data):
        st.warning("UnstructuredURLLoader failed to load data. Trying alternative method.")
        data = []
        for url in urls:
            document = alternative_data_loading(url)
            if document:
                data.append(document)
            else:
                st.error(f"Failed to load content from {url}. Please check the URL.")
                st.stop()

    # Split data into smaller chunks using RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    main_placeholder.text("🔄 Splitting text into smaller chunks...")
    docs = text_splitter.split_documents(data)

    if not docs:
        st.error("Failed to split documents. Please check the data content.")
        st.stop()

    # Create embeddings using OpenAI's Embeddings API
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    embeddings_list = embeddings.embed_documents([doc.page_content for doc in docs])

    if not embeddings_list:
        st.error("Failed to generate embeddings. Please check your API key and document content.")
        st.stop()

    # Store embeddings in FAISS vectorstore
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("🔄 Building the FAISS index...")
    time.sleep(2)

    # Save the FAISS index and metadata
    faiss.write_index(vectorstore_openai.index, index_file_path)
    metadata = {
        "docstore": vectorstore_openai.docstore,
        "index_to_docstore_id": vectorstore_openai.index_to_docstore_id
    }
    with open(metadata_file_path, "wb") as f:
        pickle.dump(metadata, f)

# Input for user query
query = main_placeholder.text_input("🔎 Enter your query: ", placeholder="Ask a question based on the articles processed")
if query:
    if os.path.exists(index_file_path) and os.path.exists(metadata_file_path):
        # Load the FAISS index from saved files
        loaded_index = faiss.read_index(index_file_path)

        # Load the metadata
        with open(metadata_file_path, "rb") as f:
            metadata = pickle.load(f)

        # Recreate the FAISS vectorstore
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS(
            index=loaded_index,
            embedding_function=embeddings.embed_query,
            docstore=metadata["docstore"],
            index_to_docstore_id=metadata["index_to_docstore_id"]
        )

        # Create the retrieval chain for answering questions
        with st.spinner("🔄 Searching for answers..."):
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

        # Display the answer and sources in columns for better layout
        col1, col2 = st.columns([2, 1])

        with col1:
            st.header("Answer")
            st.write(result["answer"])

        with col2:
            st.subheader("Sources:")
            sources = result.get("sources", "")
            if sources:
                for source in sources.split("\n"):
                    st.write(source)

