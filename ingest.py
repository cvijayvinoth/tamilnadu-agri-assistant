import os
import glob
from pypdf import PdfReader
from llm import get_embedding
from store import FaissVectorStore
from config import RAW_DATA_DIR

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 350, overlap: int = 50) -> list[dict]:
    reader = PdfReader(pdf_path)
    chunks = []
    
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_str = text[start:end]
            if chunk_str.strip():
                chunks.append({
                    "text": chunk_str.strip(),
                    "source": os.path.basename(pdf_path),
                    "page": page_num
                })
            start += (chunk_size - overlap)
            
    return chunks

def run_ingestion(data_dir: str = RAW_DATA_DIR):
    store = FaissVectorStore()
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {data_dir}.")
        return

    all_chunks = []
    for pdf in pdf_files:
        print(f"Extracting: {pdf}")
        chunks = load_and_chunk_pdf(pdf)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No valid text extracted.")
        return

    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    vectors = []
    metadata_list = []

    for idx, item in enumerate(all_chunks):
        vec = get_embedding(item["text"])
        vectors.append(vec)
        metadata_list.append({
            "text": item["text"],
            "source": item["source"],
            "page": item["page"]
        })

    store.add_vectors(vectors, metadata_list)
    print("Ingestion complete! Index saved successfully.")

if __name__ == "__main__":
    run_ingestion()
