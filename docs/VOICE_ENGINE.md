# N.I.M.R.A.S. Voice Engine (NVE)

NVE is the voice interface for N.I.M.R.A.S. You can speak English, Sinhala, or a mix of the two. It recognises what you said, keeps the conversation context, reasons with Ollama, and answers out loud. It runs **entirely on your machine with free, open-source components**: no paid APIs and no cloud services.

```
Mic ─► AudioWorklet (16 kHz) ─► browser AEC / noise suppression / AGC ─► client VAD + endpointing
   ─► WebSocket ─► VAD check/trim ─► DC + high-pass + normalise ─► language ID (Whisper, en|si)
   ─► Whisper small (English) │ SinhaSpeech Whisper (Sinhala) ─► ICT normalisation (reported)
   ─► language decision ─► local intents ─► context + RAG + memory ─► Ollama (streaming)
   ─► sentence split ─► language-aware TTS router (Piper EN / Piper Dilu SI) ─► WAV chunks
   ─► WebSocket ─► Web Audio (gain ─► analyser ─► speakers)          ◄── barge-in cancels everything
```

---

## Stage 1: Research (what was chosen and why)

| Need | Candidates evaluated | Chosen | Reason |
|---|---|---|---|
| STT engine | OpenAI Whisper (PyTorch), faster-whisper (CTranslate2), whisper.cpp | **whisper.cpp b5454** (MIT) | There are no PyTorch, CTranslate2 or onnxruntime wheels for Python 3.15. whisper.cpp ships a CPU build for Windows, `whisper-server` keeps the model loaded, and `whisper-quantize` is included. |
| English / language-ID model | tiny, base, small, medium, large-v3-turbo | **ggml-small-q5_1** (182 MB) | Measured on this CPU: about 1.8 s per 5 s clip with an adaptive audio window. Medium and large are 3 to 10 times slower without a GPU. |
| Sinhala STT | stock Whisper small; community fine-tunes (seniruk 29% WER, janiduchamika 25%, Lingalingeswaran 46%, SinhaSpeech run12 **15.5% WER / 4.4% CER**) | **SinhaSpeech/whisper-small-sinhala-run12-best** (Apache-2.0) | Stock Whisper produced **degenerate repeated output on Sinhala** (measured). SinhaSpeech was best on a test set of 15,860 utterances from speakers not seen in training. It is converted locally to ggml by `voice/convert_whisper.py`, which needs only numpy. |
| English TTS | Piper, Windows SAPI, browser speechSynthesis | **Piper en_US-lessac-medium**, with SAPI as an offline fallback | Real-time factor of about 0.1 on CPU, natural voice, MIT engine. |
| Sinhala TTS | Dilu Piper voice, MMS-TTS (needs PyTorch) | **Dilu si_LK-dilu-medium** | Runs in the same Piper binary. espeak-ng's Sinhala rules read digits correctly (25 → "visipəhə"). |
| Reasoning | llama3.2:1b, gemma3:4b | **gemma3:4b for voice** (`voice.ollama_model`); chat and learning keep their own model | llama3.2:1b produced unusable, repetitive Sinhala (measured). Gemma 3 is trained on more than 140 languages. |
| Transport | WebSocket, HTTP chunked, WebRTC | **WebSocket** (stdlib RFC 6455) | Two-way, low latency, and no extra dependencies. |

**Deviation from the preferred stack (React/TypeScript/Tailwind + FastAPI):** the existing N.I.M.R.A.S. code is stdlib-only Python 3.15 with vanilla JS. FastAPI's pydantic-core has no Python 3.15 wheel. React would add a Node build step to an app that currently has none. NVE therefore uses the same stdlib server, a minimal WebSocket implementation and plain ES2020 + Web Audio. The engine core (`nimras/voice/engine.py`) does not depend on the transport, so a FastAPI or React front end can be added later without touching it.

## Stage 2: Architecture

| File | Role |
|---|---|
| `nimras/voice/engine.py` | `VoiceEngine` (shared models) and `VoiceSession` (one conversation): the pipeline, cancellation, metrics |
| `nimras/voice/stt.py` | `STTProvider` interface; `BilingualWhisper` (language ID, then the per-language model); `WhisperModel` (whisper-server with CLI fallback) |
| `nimras/voice/tts.py` | `Voice` interface; `PiperVoice` (persistent process); `SapiVoice`; `TTSRouter` (mixed-language splitting) |
| `nimras/voice/language.py` | Script and Singlish detection, explicit language requests, reply-language decision, TTS segmentation, sentence splitting |
| `nimras/voice/vocabulary.py` | ICT recognition hints, conservative normalisation, pronunciation lexicon |
| `nimras/voice/audio.py` | WAV I/O, resampling, preprocessing, energy VAD, test-signal helpers |
| `nimras/voice/ws.py`, `transport.py` | WebSocket protocol (Origin-checked) and message handling |
| `nimras/voice/selftest.py` | Automated tests: WER, CER, latency, failure injection, resources |
| `nimras/voice/registry.py` | Model registry, benchmark gate, user-authorised promotion, rollback, release-feed watching |
| `nimras/voice/assets.py`, `convert_whisper.py` | Pinned, hash-recorded downloads; Hugging Face to ggml conversion and quantisation |
| `nimras/voice/hardware.py` | Hardware inspection and model recommendation |
| `nimras/voice/api.py` | HTTP routes under `/api/voice/*` |
| `nimras/web/voice.js`, `voice-worklet.js`, `voice.css` | Voice-first UI, capture worklet, visualisation |

Every speech engine sits behind an interface (`STTProvider`, `Voice`) selected in config, so any of them can be replaced without rebuilding the app.

## Installation

```powershell
python -m nimras voice setup      # about 600 MB: whisper.cpp, Piper, Whisper small, EN and SI voices, Sinhala STT (converted locally)
ollama pull gemma3:4b             # multilingual reasoning model for voice (3.3 GB)
python -m nimras serve            # http://127.0.0.1:8765 opens on the Voice tab
```

`voice setup` is the **only** code path that downloads speech software, and only when you run it. Each file's SHA-256 is written to `voice_models/manifest.lock.json`, and Hugging Face files are checked against the publisher's LFS hash. The autonomous learner never downloads or runs software.

Optional larger models: `python -m nimras voice setup --with stt_whisper_medium_q5` (or `stt_whisper_large_v3_turbo_q5` for a CUDA machine).

## Stage 3: Microphone and audio capture
- The microphone is opened **only** when you press 🎤, using `getUserMedia` with `echoCancellation`, `noiseSuppression` and `autoGainControl` (WebRTC's built-in, local processing).
- `voice-worklet.js` runs on the audio thread. It downsamples to 16 kHz and emits 20 ms Int16 frames with their RMS level.
- Audio is streamed **only while an utterance is open**. Nothing is sent while you are silent.
- While the mic is on, a red **"MIC LIVE"** indicator shows. Pressing 🎤 again stops every track, so the browser's own mic indicator also turns off.

## Stage 4: Speech recognition
- Two `whisper-server` processes run on loopback ports 8771 and 8772, so the models are loaded once.
- An **adaptive audio window** (`audio_ctx = max(448, 50·s + 160)`) is used. Measured on this machine: a 5.4 s English clip took 1.8 s instead of 6.2 s, with an identical transcript.
- The Sinhala model gets no prompt, because it was fine-tuned without prompts. The English model gets the ICT prompt.

## Stage 5: Language detection and Sinhala-English handling
- **Acoustic language ID:** Whisper's detect-only pass, with probabilities limited to `voice.stt.allowed_languages` (en, si). Sri Lankan speech labelled as Hindi or Tamil is mapped back.
- **Text analysis:** the Sinhala script ratio, plus a Singlish lexicon for Sinhala written in Latin letters, gives `en`, `si` or `mixed`.
- **Reply language:** an explicit request in the current utterance ("explain that in Sinhala", "සිංහලෙන් කියන්න") wins. Otherwise a preference the user set explicitly earlier applies (sticky). Otherwise N.I.M.R.A.S. replies in the language spoken; mixed speech gets a Sinhala reply.
- **ICT normalisation** changes only case, spacing and punctuation of known terms ("v lan" → VLAN, "d h c p" → DHCP, "hyper v" → Hyper-V). Letters are never changed, and every change is shown in the UI ("normalised: dhcp → DHCP").

## Stage 6: Connection to Ollama
- `ChatService.build_context` is shared with text chat, so voice uses the same RAG knowledge, approved personal memory and conversation history.
- A **voice style** instruction is added: no markdown, short answers unless detail is asked for, no re-introductions, ask a clarifying question when the request is unclear, follow corrections, and expect recognition errors.
- Replies stream (`OllamaClient.chat_stream`). While a voice turn is active, background learning **holds its model calls** (`OllamaClient.interactive()`).

## Stage 7: English and Sinhala speech generation
- Each reply is split into sentences. The first clause is spoken early to reduce the time to first audio.
- `TTSRouter` splits mixed text into script runs ("ඔබේ │firewall│ එකේ │VLAN 10│ …"). Each run goes to its voice, edge silence is trimmed, and runs are joined with 60 ms gaps and 12 ms fades.
- The English pronunciation lexicon handles terms like SIEM → "seem", VLAN → "V-LAN", ITIL → "eye-till". URLs, markdown and code blocks are not read aloud.
- Speech rate (Piper `length_scale`), volume (Web Audio gain) and per-language voice selection are adjustable.

## Stage 8: Streaming audio and interruption
- Audio chunks are `[u32 turn][u32 seq][WAV]` binary frames, decoded in order and scheduled back-to-back so there are no gaps.
- **Barge-in:** while N.I.M.R.A.S. speaks, the client VAD uses a stricter threshold, `max(start × barge_in_ratio, AI output level × 0.9)`, and needs at least `barge_in_ms` of speech. This is the echo guard on top of browser echo cancellation. When it triggers:
  1. Playback stops locally.
  2. The interrupted turn's chunks are discarded.
  3. `interrupt` is sent; the server cancels the LLM stream and pending TTS.
  4. The partial answer is stored as `… [interrupted by user]`, so the next turn can refer to it ("explain *that* in Sinhala").
- The ■ Stop button and the spoken words "stop" or "නවත්තන්න" also stop the current answer.

## Stage 9: Voice-first dashboard
The **Voice** tab is now the landing page. It has:
- a central orb whose radius and radial spectrum follow the live mic or AI analyser;
- separate canvases for the user's waveform and the AI's waveform, with level meters;
- the visual states IDLE, LISTENING, PROCESSING, SPEAKING, INTERRUPTED, ERROR and OFFLINE;
- the mic toggle, Stop, the heard and replying languages with probabilities, captions (not chat bubbles), connection and engine status, voice settings, and per-turn latency.

Rendering pauses when the tab is hidden and drops to about 12 fps when nothing is active. The **Voice lab** tab shows hardware, engine status, the self-test, live performance, the model registry and the privacy controls. Text chat remains available as a secondary tab, and there is a "⌨ Type instead" fallback for when the mic is unavailable.

## Stage 10: Privacy, security and error handling
- All processing is local. Speech binaries are bound to 127.0.0.1. The earlier browser speech recognition, which sends audio to Google, was **removed**.
- **Raw audio is never stored.** One recording is kept only when you press "Save last utterance to test set", correct its transcript and tick a consent box. It is stored locally for evaluation and never used for training.
- **Transcripts** are kept in conversation history so follow-up questions work. You can delete voice transcripts, all history, or the test recordings in Voice lab. Setting `voice.store_transcripts` to false keeps transcripts out of history.
- **WebSocket security:** the Origin must be this server (browsers don't apply CORS to WebSockets), together with the existing Host check. Client frames must be masked and are capped at 2 MB.
- **Page restrictions:** `Permissions-Policy: microphone=(self)`, and the CSP allows WebSockets only to this host.
- **Errors** (mic denied or unplugged, device change, STT, Ollama or TTS failure, connection lost) put the UI into the ERROR state with a readable message. The WebSocket reconnects with backoff.
- No voice cloning is implemented.

## Stage 11: Testing
```powershell
python -m unittest tests.test_voice -v        # 23 offline tests (fake engines): pipeline, interruption, language switch, security
python -m nimras voice test                   # real models: synthetic, tts, recorded, pipeline suites
python -m nimras voice transcribe clip.wav    # one file
python -m nimras voice say "ඔබේ firewall එක පරීක්ෂා කරන්න"
```
Self-test evidence is labelled honestly:
- **synthetic:** Piper → Whisper round trips, including clean, ±speed, 20 and 10 dB noise, and a quiet speaker. This is an upper bound; real voices are harder.
- **recorded:** your consented recordings. **Sri Lankan English pronunciation and Sinhala accent coverage can only be measured here.** Until such recordings exist, those checks report "needs recordings" instead of passing.
- **pipeline:** follow-up context, Sinhala reply, interruption and language switch, plus failure injection (Ollama down, TTS voice missing, STT model missing).

The results of the latest run are in `docs/VOICE_TEST_RESULTS.md`.

## Stage 12: Optimisation and deployment notes
- **CPU-only defaults** for this machine (Core Ultra 5 225U, 14 threads, 31 GB RAM, Intel graphics): Whisper small q5_1 ×2, Piper medium, gemma3:4b. Thread count is automatic (CPU − 2, max 8).
- With an NVIDIA GPU, use the CUDA build of whisper.cpp and `large-v3-turbo`.
- **Tunable** in Settings or `nimras.config.json` under `voice.*`:
  - `vad.endpoint_ms` (750)
  - `vad.start_ratio`
  - `vad.barge_in_ratio`
  - `vad.max_utterance_s`
  - `tts.speed`
  - `stt.beam_size`
  - `stt.adaptive_audio_ctx`
- **Improvement loop:** speech-model release feeds (whisper.cpp, Piper) are checked daily, and research topics (Sinhala ASR, Sinhala TTS) are queued for the learner. Recommendations appear in Voice lab. You install a candidate, `voice register` it, and the **benchmark** compares it with production. **Promotion** requires a pass *and* your confirmation, the config and DB are backed up first, and `voice rollback` restores the previous model.

## Known limitations
- **Sinhala accuracy:** the fine-tuned model reports 15.5% WER on its own test set. Real-world Sri Lankan accents, background noise and fast speech will be worse, and this has not yet been measured on real voices. Save consented recordings to measure it.
- **Code-switching:** the Sinhala STT model was trained on Sinhala. English words inside Sinhala speech may come out in Sinhala script (phonetically) rather than Latin letters.
- **Latency on this CPU:**
  - Speech recognition is about 1.5 to 3 s per utterance.
  - Gemma 3 4B on CPU adds several seconds before the first token. A GPU or a smaller multilingual model would reduce this.
  - Measured numbers are in the test results.
- **Echo:** handling depends on Chrome's echo cancellation plus the stricter barge-in threshold. Laptop speakers at high volume can still trigger false barge-ins. Headphones are recommended, or raise `voice.vad.barge_in_ratio`.
- **Interruption:** an interrupt during the LLM's prompt-processing phase takes effect when the first token arrives. Playback itself stops immediately in the browser.
- **Voices:** Dilu is a single female Sinhala voice. The router switches between the Dilu and Lessac voices mid-sentence for English terms, which is intelligible but noticeable.
- **Mic and speaker disconnection** are handled in the browser and need manual verification (unplug the device).
