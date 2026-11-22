# N.I.M.R.A.S. V2.0: Autonomous Self-Learning Intelligence

N.I.M.R.A.S. is a local AI system that keeps learning in the background. It discovers new topics on the internet, researches them, verifies what it finds against independent sources, stores the results in long-term memory, and measures whether it is improving. **Ollama** is the central inference engine. The whole core uses only the Python standard library (numpy is optional and used for faster vector search if installed).

## Quick start

```powershell
ollama pull llama3.2            # chat + analysis model (any Ollama chat model works)
ollama pull nomic-embed-text    # embeddings for semantic search / RAG
python -m nimras serve          # Learning Centre at http://127.0.0.1:8765
```

The first learning session starts about 30 seconds after launch, and then runs every `scheduler.interval_minutes`.

### Voice (English · සිංහල · mixed)

```powershell
python -m nimras voice setup      # whisper.cpp + Piper + Whisper/Sinhala STT + EN/SI voices (local, free, open source)
ollama pull gemma3:4b             # multilingual model for spoken replies (voice.ollama_model)
```

Open the **Voice** tab, press 🎤 and speak. To interrupt, just speak over it or press Stop. Everything runs locally. Audio is never stored unless you save a recording to the test set, and that requires your consent. See [docs/VOICE_ENGINE.md](docs/VOICE_ENGINE.md) for the architecture, and `python -m nimras voice test` for measured accuracy and latency.

Optional extras:

| Optional | Enables |
|---|---|
| `OLLAMA_API_KEY` env var | Ollama hosted **web search / web fetch** as a discovery provider |
| `internet.searxng_url` | A self-hosted SearXNG metasearch provider |
| Docker running | The **sandbox** used to test generated code examples (already detected on this machine) |
| `training.enabled: true` | The experimental fine-tuning dataset and model-versioning module |

Wikipedia, arXiv and RSS feeds work with no configuration.

## CLI

```
python -m nimras serve [--no-scheduler]     UI + autonomous background learning
python -m nimras learn --topic "Polars"     one learning session now (foreground)
python -m nimras status | evaluate | backup
python -m nimras ask "What is WebAssembly?"
python -m nimras dataset                    curate a fine-tuning dataset (opt-in)
python -m nimras model register|evaluate|promote|rollback|list
python -m unittest tests.test_nimras tests.test_voice -v     offline end-to-end tests
python -m nimras voice setup | status | hardware | test | say "…" | transcribe f.wav
python -m nimras voice models | register | evaluate --id N | promote --id N --yes | rollback
```

## Architecture

```
                 ┌────────────── LearningScheduler (background thread) ───────────────┐
                 │ quiet hours · defers while user active · per-session budgets:       │
                 │ max topics / page fetches / LLM calls / minutes · stop & pause      │
                 └──────────────────────────────┬──────────────────────────────────────┘
 topic queue  ◀── seeds · user · knowledge gaps (chat/eval) · revisits · RSS emerging-tech discovery
      │
      ▼
 DISCOVERY ─────── providers: Ollama web search, SearXNG, Wikipedia, arXiv, (DuckDuckGo opt-in)
      │            query templates + providers chosen by a UCB bandit rewarded by verified yield
      ▼
 RESEARCH ──────── Fetcher: URL/SSRF policy, robots.txt, per-domain rate limit, size cap, text only
      ▼
 EXTRACTION ────── HTML→text, publication date, exact + simhash near-duplicate removal,
      │            prompt-injection scan & neutralisation
      ▼
 ANALYSIS ──────── Ollama (fenced untrusted input): relevance, layer, category, atomic claims, related topics
      ▼
 COMPARISON ────── new claims vs existing: same → corroborate · contradicts → contradiction record · new
      ▼
 VERIFICATION ──── claim verified only with ≥2 independent credible domains, or 1 authoritative source,
      │            and no credible contradiction. Domain reputation learns from corroboration history.
      ▼
 STORAGE ───────── SQLite: entries (versioned), claims ↔ sources, archive chunks + embeddings (FTS5 + vectors)
      ▼
 DEEP STUDY ────── for software: official docs → purpose → comparison → pros/cons → example → Docker test
      ▼
 EVALUATION ────── SelfImprovementEngine (every N sessions): answer judging, repeated mistakes,
                   retrieval recall@k / MRR probes, prompt A/B experiments, duplicate merging,
                   metric deltas vs previous run
```

| Module | Responsibility |
|---|---|
| `nimras/learning/pipeline.py` | The end-to-end learning flow for each topic and source |
| `nimras/learning/discovery.py` | Search providers, RSS/Atom, the self-tuning strategy bandit |
| `nimras/learning/fetcher.py` | Policy-enforcing retrieval and extraction |
| `nimras/learning/credibility.py` | Domain priors combined with a learned track record |
| `nimras/learning/agent.py` | Deep-study research agent (the framework example from the spec) |
| `nimras/knowledge.py` | Knowledge layers, claims, verification, versions, hybrid RAG |
| `nimras/evaluation.py` | Self-improvement engine |
| `nimras/chat.py` | RAG chat; turns unanswerable questions into knowledge gaps |
| `nimras/training.py` | Experimental: datasets, candidate models, evaluation gate, rollback |
| `nimras/security.py` | Injection defence, outbound privacy, network policy |
| `nimras/sandbox.py` | Docker isolation (no network, read-only, no capabilities, limits) |
| `nimras/server.py` + `web/` | The **N.I.M.R.A.S. LEARNING CENTRE** UI and local API |

### Knowledge layers
- **Personal memory**: user preferences and approved facts. Items are stored *pending* until you approve them, are never sent to the internet, and are scrubbed from outgoing search queries.
- **General knowledge** and **technical knowledge**: verified entries, categorised (science, business, networking, cybersecurity, AI/ML, and so on).
- **Internet learning archive**: chunked text of every retained source with full provenance (URL, publication date, retrieval date, credibility).

### Truth model
Nothing from the internet is treated as fact by default. Each piece of knowledge has a status:
- `verified`: corroborated by independent credible sources.
- `provisional`: not yet corroborated.
- `contested`: credible sources disagree.
- `outdated`: superseded.

Every entry change writes a version-history row, so nothing is silently overwritten. Chat answers show the status of every knowledge item they used.

## Autonomy and security boundaries
- Web content is always **data**. It is wrapped in random delimiters it cannot forge, scanned for injection patterns, neutralised, and audited. Domains that attempt injection lose credibility.
- Only public `http(s)` hosts are reachable. Localhost, LAN and cloud-metadata IPs are blocked, and the check is repeated on every redirect.
- **Only text content types are accepted. Downloaded software is never executed.** The only code that ever runs is short model-written examples, inside Docker with `--network none`. If Docker is unavailable, the code is not run.
- No OS settings are changed and no private accounts are accessed. The UI cannot edit security settings (domain lists, sandbox, model promotion); those live only in `nimras.config.json`.
- The production model is never replaced without evaluation. Promotion is refused unless the candidate beats production on held-out data by `training.promotion_margin`, and rollback is one command.
- Backups of the database and config are made daily and on demand. Every autonomous action is written to the **audit trail**.
- The local API binds to `127.0.0.1` and has DNS-rebinding and CSRF protection.

## Honest limitations
- **Learning updates the knowledge base, not the model's weights.** Fine-tuning is prepared here (curated, quality-checked datasets), but the training itself runs in an external tool such as Unsloth, Axolotl or LLaMA-Factory; see `data/training/*/README.md`. Expect it to need roughly 200 or more good examples before it helps.
- Verification is corroboration-based. It rejects weakly sourced or contradicted claims, but it cannot guarantee truth if several credible sources repeat the same error.
- Analysis quality scales with the Ollama model you choose. Small models extract fewer and noisier claims. You can set `ollama.analysis_model` to a larger model for learning while keeping a fast chat model.
- The sandbox runs Python examples that use only the standard library. Examples that need third-party packages are stored but marked `not_executed`, because installing downloaded packages is deliberately disabled.
