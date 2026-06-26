# app.py
import streamlit as st
import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="RAG: FYP Handbook Assistant", page_icon="📚")
st.title("🤖 RAG: FYP Handbook Assistant")
st.write("Ask any question regarding the FAST-NUCES BS Final Year Project process.")

@st.cache_resource
def load_resources():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("faiss_handbook.index")
    with open("metadata.pkl", "rb") as f:
        chunks_text, chunks_metadata = pickle.load(f)
    return model, index, chunks_text, chunks_metadata

# Verify if index files exist on disk before launching the interface
if not (os.path.exists("faiss_handbook.index") and os.path.exists("metadata.pkl")):
    st.error("⚠️ Missing Local Databases! Please run 'python ingest.py' in your terminal first to process the PDF.")
else:
    model, index, chunks_text, chunks_metadata = load_resources()

    with st.sidebar:
        st.header("⚙️ Pipeline Configuration")
        st.success("✅ FAISS Database Connected")
        st.info("Embedding Model:\n`all-MiniLM-L6-v2`\n\nSimilarity Threshold:\n`0.25`\n\nTop-K Matches:\n`5`")

    user_question = st.text_input("Enter your handbook query here:", placeholder="e.g., What margins and spacing do we use?")

    if st.button("Ask Assistant") and user_question.strip():
        # Embed user input query
        query_vector = model.encode([user_question])
        query_vector_np = np.array(query_vector).astype("float32")
        faiss.normalize_L2(query_vector_np)
        
        # Execute k-nearest search via FAISS (k=5)
        distances, indices = index.search(query_vector_np, k=5)
        top_score = distances[0][0]
        
        # Check strict prompt threshold parameter limit (0.25)
        if top_score < 0.25:
            st.error("⚠️ System Guardrail: I don't have that specific rule in the handbook context.")
        else:
            st.subheader("📝 Grounded Context Answer")
            best_idx = indices[0][0]
            best_meta = chunks_metadata[best_idx]
            
            # Print the most accurate grounded context paragraph from the PDF
            st.info(f"**Answer matched from (p. {best_meta['page']}) - Section: {best_meta['section_hint']}:**")
            st.write(chunks_text[best_idx])
            
            # Print the collapsible sources list tracking chunk details
            with st.expander("🔍 Collapsible Sources Debug Block (Top-K Chunks)"):
                for score, idx in zip(distances[0], indices[0]):
                    if idx != -1:
                        meta = chunks_metadata[idx]
                        st.markdown(f"- **Page {meta['page']}** ({meta['section_hint']}) — Vector Match Score: `{score:.3f}`")
