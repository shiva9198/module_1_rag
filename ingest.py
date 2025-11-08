import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Define the path for the data and the vector store
DATA_PATH = "data/sample_doc.txt"
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    """
    Creates and saves a FAISS vector database from a text document.
    """
    
    # 1. Load the document
    # We use TextLoader for .txt files
    print(f"Loading document from {DATA_PATH}...")
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()
    print("Document loaded successfully.")

    # 2. Split the document into chunks
    # This splitter is good for general text.
    # It tries to split on common separators like newlines and spaces.
    print("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Document split into {len(chunks)} chunks.")

    # 3. Choose an embedding model
    # We're using a popular, fast, and local model from Hugging Face.
    # 'all-MiniLM-L6-v2' is a great starting point.
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    print("Embedding model initialized.")

    # 4. Create the FAISS vector store
    # This process takes the text chunks and the embedding model
    # and creates the vector representations.
    print("Creating FAISS vector store from chunks...")
    db = FAISS.from_documents(chunks, embeddings)
    print("Vector store created successfully.")

    # 5. Save the vector store locally
    # We create a new folder 'vectorstore' to hold the index
    if not os.path.exists(DB_FAISS_PATH):
        os.makedirs(DB_FAISS_PATH)
        
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store saved to {DB_FAISS_PATH}.")

if __name__ == "__main__":
    create_vector_db()
