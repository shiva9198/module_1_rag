import os
import argparse
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- CONFIGURATION ---
load_dotenv()  # Load variables from .env file (OPENROUTER_API_KEY)

# Ensure the API key is set
if "OPENROUTER_API_KEY" not in os.environ:
    raise EnvironmentError("OPENROUTER_API_KEY not found in .env file. "
                           "Please add it and try again.")

DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"  # Free model on OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- CORE LOGIC ---

def load_components():
    """
    Initializes and returns the core components for the RAG chain:
    LLM, Embeddings, Vector Store, and Retriever.
    """
    print("Loading components...")
    
    # 1. Initialize LLM (OpenRouter)
    # We use Llama 3.1 8B via OpenRouter for free access
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.2,
        openai_api_key=os.environ["OPENROUTER_API_KEY"],
        openai_api_base=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "http://localhost:3000",  # Your app URL
            "X-Title": "RAG Assistant",  # Your app name
        }
    )
    
    # 2. Initialize Embedding Model (HuggingFace)
    # Must be the *same model* used in ingest.py
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                       model_kwargs={'device': 'cpu'})
    
    # 3. Load the FAISS Vector Store
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Vector store not found at {DB_FAISS_PATH}. "
                               f"Please run ingest.py first.")
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    # 4. Create the Retriever
    # This component "retrieves" documents from the vector store
    retriever = db.as_retriever(search_kwargs={'k': 2}) # Retrieve top 2 chunks
    
    print("Components loaded successfully.")
    return llm, retriever

def create_rag_chain(llm, retriever):
    """
    Creates and returns the RAG (Retrieval-Augmented Generation) chain.
    """
    print("Creating RAG chain...")

    # 1. Define the System Prompt
    # This is our AI's "Operating Manual" (Week 3, Lesson 2)
    # It instructs the LLM *how* to use the provided context.
    prompt_template = """
    You are an assistant for question-answering tasks. 
    Use the following retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. 
    Be concise and answer in 2-3 sentences.

    CONTEXT: 
    {context} 

    QUESTION: 
    {input} 

    ANSWER:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 2. Create the "Stuff" Chain
    # This chain "stuffs" the retrieved documents into the prompt
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 3. Create the Retrieval Chain
    # This chain:
    #   1. Takes the user's input (question)
    #   2. Passes it to the retriever to get relevant documents
    #   3. Passes the documents and the input to the question_answer_chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("RAG chain created.")
    return rag_chain

def run_query(chain, query):
    """
    Runs a query through the RAG chain and prints the results.
    """
    print(f"\nProcessing query: '{query}'")

    # The chain created by `create_retrieval_chain` will internally call the
    # retriever, but we also perform an explicit retrieval here so we can
    # surface which document chunks are being used and handle empty results.
    docs = []
    try:
        # Many LangChain retrievers implement `get_relevant_documents`
        if hasattr(chain, "retriever") and hasattr(chain.retriever, "get_relevant_documents"):
            docs = chain.retriever.get_relevant_documents(query)
        elif hasattr(chain, "retriever") and hasattr(chain.retriever, "retrieve"):
            docs = chain.retriever.retrieve(query)
    except Exception:
        docs = []

    if not docs:
        print("\n[WARN] No relevant context found in the vector store for this query.")
        print("Try rephrasing the question or run `python ingest.py` with more documents.")

    else:
        print("\n--- RETRIEVED CONTEXT (preview) ---")
        for i, d in enumerate(docs[:3]):
            snippet = d.page_content[:300].replace('\n', ' ')
            print(f"Chunk {i+1}: {snippet}...\n")

    # Invoke the chain (the chain will still use its configured retriever)
    try:
        result = chain.invoke({"input": query})
    except Exception as e:
        print(f"[ERROR] Failed to invoke RAG chain: {e}")
        return

    print("\n--- ANSWER ---")
    # The chain returns the answer under the `answer` key in this setup
    answer = result.get("answer") if isinstance(result, dict) else result
    print(answer)

# --- MAIN EXECUTION ---

def main():
    """
    Main function to run the RAG assistant.
    Supports running a single query from the command line
    or entering an interactive loop.
    """
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the RAG assistant.")
    parser.add_argument("query", 
                        type=str, 
                        nargs='?',  # Makes the argument optional
                        default=None, 
                        help="A single query to ask the assistant.")
    args = parser.parse_args()
    
    # Load components and create the chain
    llm, retriever = load_components()
    rag_chain = create_rag_chain(llm, retriever)
    
    if args.query:
        # If a query was provided as an argument, run it and exit
        run_query(rag_chain, args.query)
    else:
        # Otherwise, start an interactive loop
        print("\n--- RAG Assistant ---")
        print("Enter your query. Type 'exit' or 'quit' to stop.")
        
        while True:
            try:
                user_query = input("\nQuery: ")
                if user_query.lower() in ['exit', 'quit']:
                    print("Exiting...")
                    break
                if user_query.strip() == "":
                    continue
                
                run_query(rag_chain, user_query)
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break

if __name__ == "__main__":
    main()