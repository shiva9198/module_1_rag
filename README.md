# Module 1 Project: Local RAG Assistant

This project is a complete, locally-run Retrieval-Augmented Generation (RAG) assistant built for the Agentic AI Developer Certification (AAIDC).

It uses a local vector store (FAISS) and local embedding models (from Hugging Face) to answer questions based on a provided text corpus. The final answer generation is powered by OpenRouter LLMs via LangChain.

## 🏛️ Project Architecture

This RAG pipeline follows the core architecture discussed in **Week 3** of the curriculum.

The flow is as follows:

1.  **Ingestion (`ingest.py`):**
    * **Load:** A source document (`data/sample_doc.txt`) is loaded.
    * **Chunk:** The document is split into smaller, manageable chunks (`RecursiveCharacterTextSplitter`).
    * **Embed:** Each chunk is converted into a vector embedding using a local model (`all-MiniLM-L6-v2`).
    * **Store:** These embeddings are saved in a local FAISS vector database (`vectorstore/db_faiss`).

2.  **Retrieval & Generation (`main.py`):**
    * **User Query:** The user provides a question (e.g., "What is agentic AI?").
    * **Retrieve:** The user's query is embedded, and FAISS is searched for the top-k (k=2) most similar document chunks (the "context").
    * **Prompt:** The retrieved context and the original query are inserted into a System Prompt Template (see **Week 3, Lesson 2**).
    * **Generate:** The complete prompt is sent to the OpenRouter LLM (`meta-llama/llama-3.1-8b-instruct`), which generates a grounded answer based *only* on the provided context.

## 🚀 Setup & Installation

Follow these steps to run the project locally.

**1. Clone the Repository:**
```bash
git clone <your-github-repo-url>
cd module_1_rag
```

**2. Create and Activate a Virtual Environment:**
```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set Up Environment Variables:**
* Create an OpenRouter API Key at [OpenRouter Keys](https://openrouter.ai/keys).
* Copy the `.env_example` file to a new `.env` file:
    ```bash
    cp .env_example .env
    ```
* Open the `.env` file and paste your OpenRouter API key:
    ```
    OPENROUTER_API_KEY="sk-or-v1-YourKeyHere..."
    ```
This file is listed in `.gitignore` and will not be committed to GitHub.

## 🛠️ Usage

There are two steps to run the application.

**Step 1: Ingest Your Data**
First, run the ingestion script to create the vector store.
(Make sure you have added your document(s) to the `data/` folder).

```bash
python ingest.py
```
**Output:**
```
Loading document from data/sample_doc.txt...
Document loaded successfully.
Splitting document into chunks...
Document split into 1 chunks.
Initializing embedding model...
Embedding model initialized.
Creating FAISS vector store from chunks...
Vector store created successfully.
Vector store saved to vectorstore/db_faiss.
```

**Step 2: Run the RAG Assistant**
You can run the assistant in two modes.

* **Interactive Mode:**
    ```bash
    python main.py
    ```
    **Output:**
    ```
    Loading components...
    Components loaded successfully.
    Creating RAG chain...
    RAG chain created.

    --- RAG Assistant ---
    Enter your query. Type 'exit' or 'quit' to stop.

    Query:
    ```

* **Single Query Mode:**
    ```bash
    python main.py "Your question here"
    ```

## 📋 Sample Input/Output

**Query:**
```
What is agentic AI?
```

**Response:**
```
--- ANSWER ---
Agentic AI systems are defined by their ability to perceive context, plan actions, and use tools to achieve goals. They are particularly beneficial for dynamic tasks where information may be incomplete, offering a more adaptive approach than classic pipelines.
```

**Query:**
```
What are the core components of an AI agent?
```

**Response:**
```
--- ANSWER ---
The core components of AI agents include goals (which define their objectives), memory for context, access to tools (like APIs), and planning policies to make decisions.
```

## 📂 Project Structure
```
module_1_rag/
├── data/
│   └── sample_doc.txt    # Your source document(s)
├── vectorstore/
│   └── db_faiss/         # Our saved FAISS index (created by ingest.py)
│       ├── index.faiss
│       └── index.pkl
├── .vscode/
│   └── settings.json     # VS Code workspace settings
├── .env                  # Secret keys (e.g., OPENROUTER_API_KEY)
├── .env_example          # Template for .env
├── .gitignore            # Files to ignore (e.g., venv/, .env)
├── ingest.py             # Script to load, chunk, embed, and save data
├── main.py               # The main RAG application script
├── README.md             # This file
└── requirements.txt      # Project dependencies
```

## 🧠 Prompts (AAIDC-Week3-Lesson-2)

The core reliability of this RAG system comes from its system prompt. The prompt explicitly instructs the LLM on its role, constraints, and how to use the provided context.

**System Prompt from `main.py`:**
```python
"""
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
```

## 🔧 Technical Configuration

**LLM Provider:** OpenRouter (https://openrouter.ai)
- **Model:** `meta-llama/llama-3.1-8b-instruct` (free tier)
- **Alternative models:** You can change the `LLM_MODEL` variable in `main.py` to use other models like:
  - `google/gemma-2-9b-it:free` (free)
  - `anthropic/claude-3-haiku` (paid)
  - `openai/gpt-4o-mini` (paid)

**Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (local, runs on CPU)

**Vector Store:** FAISS (Facebook AI Similarity Search) - local, persistent storage

## ⚖️ Evaluation & Limitations

This project successfully implements a core RAG pipeline, but it has limitations:

* **No Memory:** The assistant does *not* remember previous questions (it lacks the "Memory Management Strategies" from **AAIDC-Week3-Lesson-3a**). Each query is independent.
* **Limited Knowledge:** It can *only* answer questions based on the text in `data/sample_doc.txt`.
* **Simple Retrieval:** It uses a basic "Top-K" retrieval. It doesn't use more advanced techniques like query re-writing or reranking.
* **Static Data:** The vector store is static. If the source document changes, `ingest.py` must be run again.

## 🚧 Troubleshooting

**Common Issues:**

1. **Import errors:** Make sure you're using the virtual environment:
   ```bash
   source venv/bin/activate  # On macOS/Linux
   .\venv\Scripts\activate   # On Windows
   ```

2. **Authentication errors:** Verify your OpenRouter API key is correctly set in `.env`

3. **Vector store not found:** Run `python ingest.py` first to create the FAISS index

4. **Package not found:** Install missing dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📚 Learning Objectives Met

This project demonstrates the following AAIDC curriculum concepts:

- **Week 1, Lesson 4:** Tools of the Trade (LangChain, vector stores, embeddings)
- **Week 2:** Setup (Environment configuration, API keys, local models)
- **Week 3, Lesson 1:** LLM Integration (OpenRouter/LangChain integration)
- **Week 3, Lesson 2:** System Prompts (RAG-specific prompt engineering)
- **Week 3 Preview:** Data preparation (chunking, embedding, vector storage)

---

## 📄 License

This project is created for educational purposes as part of the Agentic AI Developer Certification program.
