# 📚 RAG: FYP Handbook Assistant

A production-ready, localized Retrieval-Augmented Generation (RAG) pipeline engineered to automate context-aware, zero-hallucination querying over the official FAST-NUCES BS Final Year Project Handbook. The system structures unstructured PDF manuals, matches semantic vectors locally, and generates precise, citation-backed answers with strict grounding constraints.

---

## 🛠️ System Architecture & Pipeline Design

The pipeline is split into isolated execution modules to optimize data ingestion and real-time inference:

```text
[BS FYP Handbook PDF] ──> [Extraction & Metadata Ingestion] 
                                       │
                                       ▼
[FAISS Index Store]   <── [Sentence-BERT: all-MiniLM-L6-v2]
        │
        ├──> [User Query] ──> [Cosine Similarity Lookup (Top-K=5)]
        │                                  │
        ▼                                  ▼
[Strict Grounding Prompt] ──> [LLM Synthesis Engine] ──> [Page-Cited Response]
```

### Infrastructure & Core Stack
- **Ingestion Pipeline (`ingest.py`):** Structured PDF parser executing page-by-page tokenization.
- **Embedding Model:** `all-MiniLM-L6-v2` (Sentence-Transformers) mapping text chunks into dense vectors.
- **Vector Database Store:** FAISS (Facebook AI Similarity Search) serialized locally to disk.
- **Inference Server (`ask.py` / `app.py`):** Streamlit graphical user interface and API router.
- **LLM Synthesis Gateway:** OpenAI GPT API / Open-Source Local Models.

---

## ✨ Engineering Implementation Details

### 1. Context-Preserving Page Chunking
- **Granular Tokenization:** Text is extracted page-by-page to preserve explicit page markers. Chunks are generated at a size of **250–400 words** with a **20%–40% sliding window overlap**.
- **Metadata Enriched Indexing:** Each generated chunk vector is tagged with an immutable metadata payload tracking:
  ```json
  {
    "chunk_id": "chunk_042",
    "page_number": "X",
    "section_hint": "First identified Section Heading"
  }
  ```

### 2. FAISS Vector Search Optimization
- Real-time user queries are dynamically vectorized using the same `all-MiniLM-L6-v2` Sentence-BERT model.
- A **Cosine Similarity** search extracts the **Top-K=5** most contextually relevant chunks within milliseconds.

### 3. Strict Semantic Grounding & Guardrails
- **Anti-Hallucination Threshold:** If the top-k similarity score falls below a baseline threshold of **0.25**, the system bypasses LLM inference entirely and securely responds: *“I don’t have that in the handbook.”*
- **Scope Restriction:** The system prompt forces strict domain isolation. It completely bars answering questions outside the scope of the provided handbook data.

---

## 🎯 Verified Validation Benchmarks

The RAG pipeline natively parses, matches, and answers complex structural manual parameters based on the following verified validation test suites:

1. **Document Formatting Configurations:** Accurately extracts report guidelines (Times New Roman 11 body fonts, Arial heading styles, and specific Title layout parameters).
2. **Page Spacing & Margin Metrics:** Successfully parses exact layout constraints (Top 1.5", Bottom 1.0", Left 2.0", Right 1.0" margins, with a 1.5 line spacing and a 6 pt paragraph rule).
3. **Development Report Formats:** Dynamically vectors the mandatory architectural chapters (Introduction, Vision, SRS, Iteration modules, and Technical User Manuals).
4. **Research & Development (R&D) Tracks:** Maps distinct academic paths requiring explicit Literature Reviews, Summaries, Critiques, and Validation/Testing steps.
5. **Academic Citation Logic:** Evaluates complex endnote rules, including the proper use parameters for linguistic markers like *“Ibid.”* and *“op. cit.”* based on text distance.
6. **Preliminary Frameworks:** Differentiates Abstract lengths (50–125 words) from complete 1–2 page Executive Summaries for project evaluations.

---

## 🚀 Repository Execution Flow

- `ingest.py`: Parses the handbook document, generates dense embedding vectors, and persists the local FAISS index file to disk.
- `ask.py`: Core inference loop handling user inputs, semantic scoring, threshold evaluations, and context compilation.
- `app.py`: Streamlit-based interface hosting the user text input, execution button, and a collapsible **Sources & Page References** debug tray.
