# ingest.py
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

print("Initializing all-MiniLM-L6-v2 embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

def run_ingestion(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found in this folder. Please add the PDF first!")
        return
        
    print(f"Parsing handbook: {pdf_path}")
    reader = PdfReader(pdf_path)
    raw_chunks = []
    metadata_list = []
    
    # Process page-by-page as required by the prompt log
    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        text = page.extract_text() or ""
        words = text.split()
        
        # Chunk settings: ~250 words per chunk with overlap
        chunk_size = 250
        overlap = 75
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 20: 
                continue
            chunk_text = " ".join(chunk_words)
            
            # Simple section parser to get context hints
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            section_hint = lines[0][:40] if lines else "General Specifications"
            
            raw_chunks.append(chunk_text)
            metadata_list.append({"page": page_num, "section_hint": section_hint})

    print(f"Generated {len(raw_chunks)} chunks. Building FAISS index...")
    embeddings = model.encode(raw_chunks, show_progress_bar=True)
    embeddings_np = np.array(embeddings).astype("float32")
    
    # Configure FAISS Index for Cosine Similarity matching
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    
    # Persist the files locally to disk
    faiss.write_index(index, "faiss_handbook.index")
    with open("metadata.pkl", "wb") as f:
        pickle.dump((raw_chunks, metadata_list), f)
    print("✅ Ingestion Complete! faiss_handbook.index and metadata.pkl saved successfully.")

if __name__ == "__main__":
    run_ingestion("BS_FYP_Handbook_2023.pdf")
