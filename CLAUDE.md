### SweetClaude

This project is managed with SweetClaude. State lives in `.sweetclaude/state/sweetclaude.yaml`; read it at session start. Use the `sweetclaude:` skills for phase work, and keep work inside the active phase and mode (`project.mode`).

### Models

- Planning, design and review of plans run in the main session on Opus (`claude-opus-5-5`).
- Implementation is delegated to Sonnet: the `sweetclaude:implementer` and `sweetclaude:test-writer` agents (already `model: sonnet`), or an Agent call with `model: "sonnet"` for other build work.

### Working conventions

- Domain terms are defined in `GLOSSARY.md`; decisions that are hard to reverse are in `docs/adr/`. Use the glossary's terms in code, docs and answers.
- Look up library and API docs with context7 before writing code against a library (FastAPI, LangGraph, Qdrant, pydantic, Next.js, …).
- Commit messages and PR descriptions carry no Claude attribution (no `Co-Authored-By: Claude`, no "Generated with Claude Code").
- Python: run project code with `.venv/Scripts/python.exe`; `python3` in Git Bash is a wrapper for the Anaconda Python, used only by SweetClaude's scripts.
