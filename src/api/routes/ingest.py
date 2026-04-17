from fastapi import APIRouter, UploadFile, File, HTTPException
from src.retrieval.chunker import load_and_chunk_bytes
from src.retrieval import vectorstore, bm25
from src.api.schemas import IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    if file.content_type not in ("application/pdf", "text/plain"):
        raise HTTPException(status_code=415, detail="Only PDF and TXT files are supported.")
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")
    try:
        chunks = load_and_chunk_bytes(content, file.filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")

    vectorstore.add_documents(chunks)
    all_docs = bm25.get_all_docs()
    bm25.build_index(all_docs + chunks)

    return IngestResponse(
        filename=file.filename,
        chunks_added=len(chunks),
        total_chunks_in_store=vectorstore.collection_count(),
    )
