# Your checklist (things only you can do)

These are the human prerequisites from the feedback (items 2–5). Work through them in one sitting; most take minutes. Tick them here so the build can rely on them.

## Hosted models: first run (OpenAI)

Chat and embeddings run on the OpenAI API and the reranker runs locally on your GPU; nothing has been called with a real key yet.

| ✓ | Task | How |
|---|---|---|
| ☐ | **Create an OpenAI API key and add credit** | platform.openai.com, API keys, then a small prepaid balance. Do not paste the key in chat. |
| ☐ | **Fill `.env`** | Copy `.env.example` to `.env` and put your key in `OPENAI_API_KEY=` (it serves chat and embeddings). `LLM_BASE_URL`, models and the rate limit are already set for OpenAI. |
| ☐ | **Check the setup** | `.venv\Scripts\python scripts\check_models.py --probe`. Every configured model must show `OK`; it also tests OpenAI embeddings at 1024 dimensions and reports whether the local reranker can run. |
| ☐ | **Start from a fresh Qdrant collection** | Embeddings are OpenAI's, so older vectors are unusable: delete the collection (or the Qdrant volume) and run `scripts\seed_corpus.py` again. |
| ☐ | **Choose the reranker** | `RERANKER_PROVIDER=local` if your laptop GPU runs the BGE reranker, else `none`. Then re-run the eval and re-tune `TAU_KEEP` / `TAU_REFUSE` (with `none` they apply to dense similarity). |
| ☐ | **Fill `MODEL_PRICES`** | Copy the current prices from openai.com/api/pricing so the dashboard shows real cost per question. |
| ☐ | **Start the stack in Docker** | Start Docker Desktop, then follow `docs/DEPLOY.md` (`docker compose --profile api up -d --build`). The images have not been built yet. |

## Today, before ingestion (blocks the MVP)

| ✓ | Task | How | Why |
|---|---|---|---|
| ☐ | **OpenAI API key in `.env`** | Copy `.env.example` to `.env` and paste the key as `OPENAI_API_KEY=sk-…`. **Don't paste it in chat**; `.env` is gitignored. | Every answer and the eval need it |
| ☐ | **Check the model ids** | `.\.venv\Scripts\python scripts\check_models.py --probe` | It shows which configured models your key can use, and whether JSON mode works. Replace any `MISSING` id in `.env` with one from `evals/configs/models_available.json`. |
| ☐ | **Install Tesseract with Urdu** | Install the Windows build of Tesseract (e.g. `winget install UB-Mannheim.TesseractOCR`), and tick **Urdu** in the language list during setup. Then `tesseract --list-langs` must show `urd` and `eng`. | Scanned PDFs, image notices and junk text layers (E1, E46, D4) |
| ☐ | **Measure OCR on real Urdu files** | `.\.venv\Scripts\python scripts\ocr_check.py "html for rag\ID-06__nadra-registration-policy-6-0-2-ur.pdf" --urdu --pages 12-14`, then the same for 1–2 scanned Urdu files from `corpus/SEED_SOURCES.md` (PB-09, RC-10) | If OCR is weak, we change the plan now (drop those sources), not on deadline day |
| ☐ | **Save the remaining blocked pages** | SP-04, SP-05, SP-06 in `corpus/MANUAL_COLLECTION.md` (PM Health steps and FAQs with every answer expanded; KP Sehat coverage) | The social-protection topic has no sources without them |
| ☐ | **Move your saved NADRA files** | Move the contents of `html for rag/` into `corpus/raw/manual/`. Both folders are gitignored. Delete the duplicate `registration-policy-6-0-2-english.pdf` (it's identical to ID-05). | The collector looks in `corpus/raw/manual/` |

## Before the eval runs (blocks M7)

| ✓ | Task | How |
|---|---|---|
| ☐ | **Review the draft golden questions** | Open `evals/golden_set.jsonl`. For each item, read the question and reference answer against the cited page. Then either set `"review_status": "approved", "reviewed_by": "<your name>"`, or fix it, or set `"dropped"`. `q_nadra_006` needs a check against the NADRA fee page first. |
| ☐ | **Write the remaining ~49 questions yourself** | Follow the subset table in `PROJECT_BLUEPRINT.md` §10.1. Priorities: Punjab-vs-Sindh differences, superseded-vs-current pairs, Urdu and Roman Urdu, 8 unanswerable. You can ask Claude for candidates, but you must check each one against the source and be able to explain it. |
| ☐ | **Label 30 answers by hand** (judge calibration) | Once the first eval run exists, use `evals/human_labels.csv` (it gets generated) |

## Before you make the repo public

| ✓ | Task | How |
|---|---|---|
| ☐ | **Verify `app/guardrails/referrals.yaml`** | For each entry, open `source_url`, confirm the number or office, then set `verified: true`. Fill in the TODO contacts or delete those entries. Unverified entries are never shown. |
| ☐ | **Send the BISP permission email** (optional) | `corpus/BISP_permission_request.md` |
| ☐ | **Re-read the "Data sources and licensing" section** in `README.md` | Make sure every agency you actually used is listed |
| ☐ | **Make the repository public** | GitHub → Settings → Danger Zone → Change visibility (the brief needs a public link) |
