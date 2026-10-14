import needle
from llm import get_embedding, generate_answer
from store import FaissVectorStore

def transcribe_audio_locally(audio_path: str) -> str:
    try:
        res = needle.transcribe(audio_path)
        return res.get("text", "").strip()
    except Exception as e:
        return f"Error transcribing audio: {e}"

def query_rag_pipeline(query_text: str, store: FaissVectorStore, max_chunks: int = 2, max_chars_per_chunk: int = 350) -> dict:
    if not query_text:
        return {"answer": "No query provided.", "sources": []}

    query_vector = get_embedding(query_text)
    matched_records = store.search(query_vector, k=max_chunks)

    compact_chunks = []
    sources = []
    for record in matched_records:
        text = record.get("text", "")
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."
        compact_chunks.append(text)
        sources.append(f"{record.get('source', 'Unknown')} (Page {record.get('page', 1)})")

    if not compact_chunks:
        return {"answer": "No relevant scheme records found in database.", "sources": []}

    answer = generate_answer(query_text, compact_chunks)
    return {"answer": answer, "sources": sources}
