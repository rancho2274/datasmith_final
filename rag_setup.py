# rag_setup.py  — RAG implementation (PDF or TXT) with LLM for answer generation
import os
from typing import Dict
from dotenv import load_dotenv
load_dotenv()

# Document loader / chunker / embeddings / vectorstore
# Prefer newer loader names if available
try:
    # LangChain v0.0x style import
    from langchain.document_loaders import PyPDFLoader, TextLoader
except Exception:
    # Fallbacks (community packages)
    try:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader
    except Exception:
        # Last resort: TextLoader only (PDF will not be supported)
        PyPDFLoader = None
        from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings and Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# sensible defaults (you can override via setup_rag_retriever params)
DEFAULT_SOURCE_FILE_TXT = "nephrology_reference.txt"
DEFAULT_SOURCE_FILE_PDF = "nephrology_reference.pdf"
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # compact and effective

RAG_LLM = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

class SimpleRetrievalChain:
    def __init__(self, vectorstore, llm, chunk_size=1000, k=3):
        self.vectorstore = vectorstore
        self.llm = llm
        self.k = k
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a clinical assistant specializing in nephrology. Answer the user's question based ONLY on the provided reference material. Always cite your sources by referencing the specific sections you used. If the reference material doesn't contain enough information to answer the question, say so clearly."),
            ("human", "Reference Material:\n{context}\n\nQuestion: {question}\n\nAnswer the question based on the reference material above. Include citations by referencing the relevant sections.")
        ])

    def invoke(self, inputs: Dict):
        query = inputs.get("input", "") if isinstance(inputs, dict) else str(inputs)
        docs = []
        try:
            if hasattr(self.vectorstore, "similarity_search"):
                docs = self.vectorstore.similarity_search(query, k=self.k)
            elif hasattr(self.vectorstore, "similarity_search_with_score"):
                docs = [d for d, _ in self.vectorstore.similarity_search_with_score(query, k=self.k)]
            elif hasattr(self.vectorstore, "as_retriever"):
                retr = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
                if hasattr(retr, "get_relevant_documents"):
                    docs = retr.get_relevant_documents(query)
                elif hasattr(retr, "get_relevant_documents_by_query"):
                    docs = retr.get_relevant_documents_by_query(query)
        except Exception:
            docs = []

        if not docs:
            return {"output": "No relevant context found in the local nephrology reference. Please try rephrasing or allow web search."}

        context_pieces = []
        for i, d in enumerate(docs, 1):
            if hasattr(d, "page_content"):
                content = d.page_content
            elif isinstance(d, dict) and "page_content" in d:
                content = d["page_content"]
            elif hasattr(d, "content"):
                content = d.content
            else:
                content = str(d)
            context_pieces.append(f"[Section {i}]\n{content[:2000]}")  # trim very long pieces in prompt

        context = "\n\n---\n\n".join(context_pieces)

        try:
            chain = self.rag_prompt | self.llm
            response = chain.invoke({"context": context, "question": query})
            if hasattr(response, "content"):
                answer = response.content
            else:
                answer = str(response)
            answer_with_citation = f"{answer}\n\n[Source: Internal Nephrology Reference - {len(docs)} section(s) retrieved]"
            return {"output": answer_with_citation}
        except Exception as e:
            return {"output": f"Retrieved context:\n{context}\n\n[Source: Internal Nephrology Reference] - (LLM generation failed with error: {e})"}

def _load_documents_from_file(source_file: str):
    ext = os.path.splitext(source_file)[1].lower()
    if ext == ".pdf":
        if 'PyPDFLoader' not in globals() or PyPDFLoader is None:
            raise ImportError("PDF loader is not available; please install a LangChain PDF loader (or place a .txt file).")
        loader = PyPDFLoader(source_file)
        documents = loader.load()
    else:
        loader = TextLoader(source_file, encoding="utf-8")
        documents = loader.load()
    return documents

def setup_rag_retriever(source_file: str = None, persist_directory: str = CHROMA_DB_PATH, rebuild: bool = False, chunk_size: int = 1000, chunk_overlap: int = 100):
    """
    Build (or load) the Chroma vectorstore and return a SimpleRetrievalChain.
    - source_file can be a .pdf or .txt path. If None: tries nephrology_reference.pdf then .txt.
    - rebuild=True forces re-ingest and overwrite of the chroma DB.
    """
    if source_file is None:
        if os.path.exists(DEFAULT_SOURCE_FILE_PDF):
            source_file = DEFAULT_SOURCE_FILE_PDF
        else:
            source_file = DEFAULT_SOURCE_FILE_TXT

    if not os.path.exists(source_file):
        raise FileNotFoundError(f"RAG source file not found: {source_file}")

    # Load documents (PDF or TXT)
    documents = _load_documents_from_file(source_file)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # Create or load Chroma vectorstore
    if os.path.exists(persist_directory) and not rebuild:
        print(f"Loading existing Chroma DB from: {persist_directory}")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("Creating Chroma vectorstore (this may take a while for a 1500-page PDF)...")
        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
        try:
            vectorstore.persist()
        except Exception:
            pass
        print(f"Vector store saved to: {persist_directory}")

    retrieval_chain = SimpleRetrievalChain(vectorstore, RAG_LLM, chunk_size=chunk_size, k=3)
    return retrieval_chain

# Do NOT run expensive setup on import.
RAG_RETRIEVAL_CHAIN = None
