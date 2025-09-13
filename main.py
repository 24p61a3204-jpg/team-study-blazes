import os
import uuid
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer
from transformers import pipeline

print("🔄 Loading Hugging Face models... this may take a minute (first run).")
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

PERSIST_DIR = "chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

app = FastAPI(title="StudyMate Backend - Hugging Face Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DOC_REGISTRY = {}

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    pages: int
    chunks: int
    message: str

class AskResponse(BaseModel):
    answer: str
    source_chunks: Optional[list] = None


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    text_parts = []
    for p in reader.pages:
        page_text = p.extract_text()
        if page_text:
            text_parts.append(page_text)
    return "\n\n".join(text_parts)

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def embed_texts(texts):
    return embedding_model.encode(texts)


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    uid = str(uuid.uuid4())
    tmp_dir = "uploads"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f"{uid}_{file.filename}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = extract_text_from_pdf(tmp_path)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from the PDF.")

    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    from langchain.embeddings.base import Embeddings

    class HFEmbeddingWrapper(Embeddings):
        def embed_documents(self, texts):
            return embedding_model.encode(texts).tolist()
        def embed_query(self, text):
            return embedding_model.encode([text])[0].tolist()

    chroma = Chroma.from_texts(chunks, embedding=HFEmbeddingWrapper(), persist_directory=PERSIST_DIR, collection_name=uid)
    chroma.persist()

    DOC_REGISTRY[uid] = {
        "filename": file.filename,
        "path": tmp_path,
        "pages": len(PdfReader(tmp_path).pages),
        "chunks": len(chunks)
    }

    return UploadResponse(
        doc_id=uid,
        filename=file.filename,
        pages=DOC_REGISTRY[uid]["pages"],
        chunks=DOC_REGISTRY[uid]["chunks"],
        message="Uploaded & indexed successfully (Hugging Face models used)."
    )


@app.post("/ask/{doc_id}", response_model=AskResponse)
async def ask_question(doc_id: str, question: str = Form(...), top_k: int = Form(4)):
    if doc_id not in DOC_REGISTRY:
        raise HTTPException(status_code=404, detail="Document not found. Upload first.")

    chroma = Chroma(persist_directory=PERSIST_DIR, collection_name=doc_id, embedding_function=lambda x: embed_texts(x).tolist())
    retriever = chroma.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    docs = retriever.get_relevant_documents(question)
    context = " ".join([d.page_content for d in docs])

    prompt = f"question: {question} context: {context}"
    result = qa_pipeline(prompt, max_length=256)
    answer = result[0]['generated_text']

    source_chunks = [{"page_content": d.page_content[:500]} for d in docs]

    return AskResponse(answer=answer, source_chunks=source_chunks)


@app.get("/docs")
def list_docs():
    return DOC_REGISTRY


@app.post("/delete/{doc_id}")
def delete_doc(doc_id: str):
    if doc_id not in DOC_REGISTRY:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        os.remove(DOC_REGISTRY[doc_id]["path"])
    except:
        pass
    del DOC_REGISTRY[doc_id]
    return {"message": "Deleted document."}
