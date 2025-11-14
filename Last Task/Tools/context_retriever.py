# tools/context_retriever.py

import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import Tool
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_PATH = "vector_store"


# 1) Read PDF file
def _read_pdf(file_path: str):
    if not os.path.exists(file_path):
        return {"status": "error", "message": f"File not found: {file_path}"}

    text = ""
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if not text.strip():
            return {"status": "error", "message": "PDF read but no text extracted."}
        return {"status": "success", "text": text.strip()}
    except Exception as e:
        return {"status": "error", "message": f"Could not read PDF: {e}"}


# 2) Embed PDF (with chunking, override previous store)
def _embed_document(file_path: str):
    pdf_result = _read_pdf(file_path)

    if pdf_result.get("status") == "error":
        return pdf_result  # {status:error, message:...}

    full_text = pdf_result["text"]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_text(full_text)

    if not chunks:
        return {
            "status": "error",
            "message": "No chunks produced from PDF text. Nothing to embed.",
        }

    # Reset vector store on each new upload (single active PDF)
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OpenAIEmbeddings()

    vector_store = Chroma(
        collection_name="pdf_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    vector_store.add_texts(chunks)
    # Since Chroma >= 0.4 auto-persists, but keep for backwards compatibility
    try:
        vector_store.persist()
    except Exception:
        pass

    return {
        "status": "success",
        "message": "PDF embedded successfully.",
        "chunks": len(chunks),
        "path": file_path,
    }


# 3) Retrieve from existing PDF vector store
def _retrieve(query: str):
    # guard for very generic questions (to avoid loops)
    TOO_GENERAL_KEYWORDS = ["content", "summary", "topics", "all"]

    q_low = query.lower().strip()
    if any(kw in q_low for kw in TOO_GENERAL_KEYWORDS):
        return "TOO_GENERAL_QUERY"

    if not os.path.exists(CHROMA_PATH):
        return {
            "status": "empty",
            "message": "No vector store found. Did you upload a PDF first?",
        }

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="pdf_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    try:
        docs = vector_store.similarity_search(q_low, k=4)
    except Exception as e:
        return {"status": "error", "message": f"Error during retrieval: {e}"}

    if not docs:
        return "NO_RELEVANT_DOCS"

    combined = "\n\n---\n\n".join([d.page_content for d in docs])

    # Return RAW TEXT so the agent can read/answer directly
    return combined


def build_retriever_tools():
    embed_tool = Tool.from_function(
        func=_embed_document,
        name="EmbedDocument",
        description=(
            "Upload & embed a PDF file into vector memory. "
            "Input MUST be a valid file path on disk, e.g. 'D:/path/to/file.pdf'."
        ),
    )

    retrieve_tool = Tool.from_function(
        func=_retrieve,
        name="RetrieveDocument",
        description=(
            "Retrieve relevant text chunks from the last uploaded PDF based on a natural-language question. "
            "ALWAYS use this for questions about 'the pdf', 'uploaded document', 'project pdf', "
            "or any follow-up related to a recently uploaded document."
        ),
    )

    return [embed_tool, retrieve_tool]
