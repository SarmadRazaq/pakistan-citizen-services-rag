# Pipeline notebooks

Step-by-step notebooks for the basic pipeline: text extraction, chunking, embedding, retrieval and
answering. They use the real project code (`app.ingestion`, `app.retrieval`, `app.generation`) and the
test fixtures (`tests/fixtures/files/factories.py`, `tests/fakes.py`). Each check ends with an `assert`, so
a notebook that runs to the end has passed. They are committed **executed, with outputs**.

| Notebook | Covers |
|---|---|
| `01_extraction.ipynb` | Validation guardrails (spoofed, empty, macro, encrypted and JavaScript files are rejected with their error codes); extraction of PDF (with a table), DOCX, PPTX, HTML, Markdown and Urdu TXT into blocks with page numbers and section paths; PDF table to Markdown; header/footer removal; a scanned PDF with and without OCR; language detection; inferred agency, jurisdiction, topic and effective date (Urdu digits included). |
| `02_chunking.ipynb` | Chunking with the configured size and overlap (512 / 64 tokens); no chunk over the limit and no text lost; section paths and page ranges; a table kept atomic with its header (and an oversized table split by rows with the header repeated); overlap and token-count histogram; configured size versus a small size, by counts only. |
| `03_embedding_and_retrieval.ipynb` | `LexicalSparseEncoder` for an English and an Urdu query (Urdu digits match ASCII digits); embedding and indexing; hybrid dense + sparse search fused with RRF, with scores; cross-variant fusion; reranking and context selection; metadata filters; tenant isolation (another user's private document is never returned); deletion removes the points. |
| `04_answering.ipynb` | The ask pipeline (LangGraph graph behind `RagAskService`) end to end: a cited answer with each citation mapped to a retrieved chunk; the retrieved chunks with scores; an Urdu question answered in Urdu; an unanswerable question refused; prompt-injection questions blocked before any work; private uploads visible to the owner only; the `outcome` field. |

`_support.py` holds the plumbing shared by the notebooks (sample documents, mode selection, the few lines
that turn chunks into indexed points). It is not a test and not part of the app.

## Offline and hosted mode

Notebooks 03 and 04 pick their models at the start and print which mode they are in.

- **Offline (default, no network, no key).** `FakeEmbedder` dense vectors, the project's
  `LexicalSparseEncoder` for the sparse half (the same composition `NvidiaEmbedder` uses), `FakeReranker`, the
  in-memory store and `FakeLLM` from `tests/fakes.py`. The banner reads
  `offline mode: fake embeddings - run with NVIDIA_API_KEY for real scores`. Scores are deterministic but are
  not semantic similarities. The fake LLM answers by quoting the best-matching source sentence, so quote
  verification and citation mapping still run for real.
- **Hosted.** When `NVIDIA_API_KEY` is set (environment variable or `.env`, read through the app's
  `Settings`; it is never printed), the notebooks use `build_retrieval(settings, qdrant_location=":memory:")`
  (NVIDIA BGE-M3 embedder and reranker, an in-memory Qdrant store) and `build_llm_clients(settings)` (hosted
  LLM chain), plus the hosted jailbreak classifier for the input guard. Notebooks 01 and 02 never need a key.

Set `RAG_NOTEBOOKS_OFFLINE=1` to force offline mode even when a key is configured. The committed outputs were
produced offline.

## Re-execute them all

From the repo root:

```
.venv/Scripts/python.exe scripts/run_notebooks.py          # all four, executed in place
.venv/Scripts/python.exe scripts/run_notebooks.py 03 04    # only those starting with 03 / 04
```

The script runs each notebook with `nbclient` (kernel `python3`, which is the interpreter running the script)
from the repo root, saves the outputs into the `.ipynb` files and exits non-zero if any cell fails. The tooling
(`nbformat`, `nbclient`, `ipykernel`) is in `requirements-dev.in` / `requirements-dev.txt`. To open them
interactively, select a Python 3 kernel from the project venv and start Jupyter in the repo root, so `import app` works.
