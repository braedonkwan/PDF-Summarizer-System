# PDF Summarizer

A local Ollama-powered PDF summarizer. It extracts text from a PDF, chunks the
text by token count, compresses each chunk with an Ollama model, recursively
compresses the result until it fits the configured context limit, then writes a
final summary to disk. Extracted pages retain `[Page N]` markers so page identity
and gaps caused by image-only pages survive later compression passes. Chunking
prefers nearby paragraph, line, and sentence boundaries while retaining a strict
token limit for long unstructured passages.

## Prerequisites

- Python 3.10 or newer
- Ollama installed and running
- An Ollama model that can handle the configured context window
- A text-based PDF. Scanned image-only PDFs need OCR before this tool can
  extract useful text.

The default config is tuned for an RTX 6000 Ada 48 GB workstation card and uses:

- Model: `igorls/gemma-4-12B-it-qat-q4_0-unquantized-heretic`
- Chunk size: `32,768` tokens
- Chunk overlap: `2,048` tokens
- Recursive compression context limit: `110,000` tokens
- Ollama context window: `131,072` tokens
- Max generated tokens per request: `16,384`
- Reserved prompt headroom: `3,000` tokens
- Less aggressive compression with `COMPRESSION_RATIO = 0.55`
- Minimum compression detail floor with `COMPRESSION_MIN_TARGET_RATIO = 0.65`
- Final summary target of `24%` of the compressed source, capped at `85%` of the generation cap
- Short-document target that starts near `50%` and smoothly transitions to the normal ratio by `8,000` source tokens
- Deterministic generation with `LLM_TEMPERATURE = 0.0` and `LLM_THINK = False`
- `LLM_KEEP_ALIVE = -1` so Ollama keeps the model loaded between requests
- Two retries with exponential backoff for transient Ollama failures
- Content-addressed response caching so interrupted runs reuse completed calls
- At most `6` recursive compression passes

This profile favors a more detailed final summary by preserving more detail
during compression, rejecting drastically undersized intermediate summaries,
and using a larger Ada-class context window for final generation. If your
machine cannot run the default model or context size, lower the model,
`CHUNK_LENGTH`, `MAX_CONTEXT_LENGTH`, `LLM_MAX_OUTPUT_TOKENS`, and
`LLM_CONTEXT_WINDOW` together in `src/pdf_summarizer/config.py`.

Every model prompt labels PDF content as untrusted source data. Embedded commands,
imitation system prompts, and requests to change the model's task are explicitly
treated as document content to summarize rather than instructions to follow.
Chunk prompts also identify their original segment number or recursive pass and
warn when context may begin or end mid-section, reducing invented transitions.

## Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows, if `python` is not on PATH but the Python launcher is available,
replace `python` with `py` in the commands above.

For VS Code Remote SSH access from Windows, create and print a local SSH key:

```powershell
C:\WINDOWS\System32\OpenSSH\ssh-keygen.exe -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519 -C "vscode-remote"

Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

On a fresh Ubuntu or WSL target, install the base packages and Ollama:

```bash
apt-get update
apt-get install -y curl ca-certificates zstd
curl -fsSL https://ollama.com/install.sh | sh
ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
```

Install and start Ollama, then pull the default model:

```powershell
ollama pull igorls/gemma-4-12B-it-qat-q4_0-unquantized-heretic
```

If Ollama is not already running, start it before running the summarizer:

```powershell
ollama serve
```

For best use of a single RTX 6000 Ada with the default profile, run Ollama with
one large model resident, one request at a time, and Flash Attention enabled:

```bash
export OLLAMA_CONTEXT_LENGTH=131072
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=f16
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
ollama serve > /tmp/ollama.log 2>&1 &
```

Confirm the model is fully on the GPU before summarizing:

```bash
ollama ps
```

The `PROCESSOR` column should show `100% GPU`.

## Configure

The constants in `src/pdf_summarizer/config.py` define the RTX 6000 Ada profile.
Input, output, model, server, and cache location can be changed per run through
command-line options without editing the profile.

| Setting | Default | Purpose |
| --- | --- | --- |
| `PDF_FILE_PATH` | `input.pdf` | PDF to summarize. Relative paths are resolved from the directory where you run the command. |
| `OUTPUT_FILE_PATH` | `summary.txt` | File written with the final summary. Parent directories are created automatically. |
| `LLM_MODEL` | `igorls/gemma-4-12B-it-qat-q4_0-unquantized-heretic` | Ollama model name. Must already be pulled locally. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `OLLAMA_PREFLIGHT` | `True` | Verify that Ollama is reachable and the configured model exists before extracting the PDF. |
| `CHUNK_LENGTH` | `32768` | Hard token ceiling for source chunks. Chunks may end slightly earlier at a nearby structural boundary. |
| `CHUNK_OVERLAP` | `2048` | Tokens repeated between adjacent chunks to preserve context. |
| `COMPRESSION_RATIO` | `0.55` | Target summary size for each compression pass. Lower values force shorter summaries; higher values preserve more detail. |
| `COMPRESSION_MIN_TARGET_RATIO` | `0.65` | Minimum useful fraction of the compression target. Summaries below this floor are retried as too short. |
| `MAX_CONTEXT_LENGTH` | `110000` | Maximum compressed text size allowed before final summarization. |
| `MAX_COMPRESSION_RETRIES` | `2` | Extra attempts when a chunk summary exceeds its target token count. |
| `LLM_MAX_OUTPUT_TOKENS` | `16384` | Maximum tokens Ollama may generate for each request. |
| `LLM_CONTEXT_WINDOW` | `131072` | Ollama `num_ctx` value. Must be larger than input plus output limits. |
| `LLM_PROMPT_RESERVE_TOKENS` | `3000` | Context space reserved for system and task instructions so source plus output cannot consume the entire window. |
| `LLM_TIMEOUT_SECONDS` | `1800.0` | HTTP timeout for each Ollama request. |
| `LLM_REQUEST_RETRIES` | `2` | Extra attempts for transport failures, HTTP 429 rate limits, and HTTP 5xx server errors. Permanent client errors fail immediately. |
| `LLM_RETRY_BACKOFF_SECONDS` | `2.0` | Initial delay before a transient retry; subsequent delays double. Set to `0` to retry immediately. |
| `LLM_KEEP_ALIVE` | `-1` | Ollama `keep_alive` value. The default keeps the model loaded for the whole run. |
| `LLM_CACHE_ENABLED` | `True` | Cache successful model responses so rerunning an interrupted job does not repeat completed calls. |
| `LLM_CACHE_DIR` | `.pdf_summarizer_cache` | Local response-cache directory. It is excluded from Git. |
| `FINAL_SUMMARY_TARGET_RATIO` | `0.24` | Final summary target as a fraction of the compressed source, capped by the final output budget. |
| `FINAL_SUMMARY_SHORT_TARGET_RATIO` | `0.5` | Detail-preserving target ratio approached for very short documents. |
| `FINAL_SUMMARY_SHORT_SOURCE_TOKENS` | `8000` | Source size at which the short-document ratio finishes transitioning to the normal final ratio. |
| `FINAL_SUMMARY_MIN_TARGET_RATIO` | `0.65` | Minimum useful fraction of the final summary target before the final pass asks for a denser version. |
| `FINAL_SUMMARY_MAX_OUTPUT_RATIO` | `0.85` | Fraction of `LLM_MAX_OUTPUT_TOKENS` used as the final summary target cap, leaving generation headroom so the model can finish naturally. |
| `FINAL_SUMMARY_EXPANSION_RETRIES` | `1` | Extra attempts when the final summary is much shorter or longer than its target range. |
| `MAX_RECURSIVE_PASSES` | `6` | Safety limit on rechunking and recompression passes. |
| `TOKEN_ENCODING` | `cl100k_base` | Tokenizer encoding used for chunking and progress counts. |

Validation runs before summarization. The tool fails fast if the PDF is missing,
if chunk settings are invalid, or if the configured prompt/output sizes cannot
fit in the Ollama context window. Blank or malformed Ollama responses are rejected.
If Ollama reports that a generation exhausted its output-token limit, the partial
text is never accepted as a finished summary; the summarizer retries with a more
concise instruction when retries remain.

## Use

After installation, summarize any PDF with:

```powershell
pdf-summarizer --input report.pdf --output report-summary.md
```

The module form remains available:

```powershell
python -m pdf_summarizer --input report.pdf --output report-summary.md
```

Useful per-run overrides include:

```powershell
pdf-summarizer `
  --input report.pdf `
  --output report-summary.md `
  --model igorls/gemma-4-12B-it-qat-q4_0-unquantized-heretic `
  --ollama-url http://localhost:11434 `
  --cache-dir .pdf_summarizer_cache
```

Use `--no-cache` for a completely fresh run without reading or writing cached
responses. Use `--skip-preflight` only when a proxy or nonstandard Ollama-compatible
server does not implement `/api/show`. Running without arguments retains the original `input.pdf` to
`summary.txt` behavior. Run `pdf-summarizer --help` for the complete option list.

Progress is printed to the console while the job runs. Example:

```text
[pdf-summarizer] Validating configuration.
[pdf-summarizer] Reading PDF: input.pdf
[pdf-summarizer] Extracted 38,421 tokens from PDF text.
[pdf-summarizer] Compression pass 1: processing 4 chunk(s).
[pdf-summarizer] Compression pass 1: chunk 1/4 (12,000 tokens).
[pdf-summarizer] pass 1, chunk 1/4: initial attempt accepted: 3,740/2,700-6,000 target tokens.
[pdf-summarizer] Compression pass 1: complete (9,832 tokens).
[pdf-summarizer] Generating final summary from 9,832 tokens.
[pdf-summarizer] Final summary generated (1,621 tokens).
[pdf-summarizer] Wrote final summary: summary.txt
```

If the original PDF already fits `MAX_CONTEXT_LENGTH`, it is sent directly to
the final synthesis so an unnecessary lossy pass cannot remove detail. Otherwise,
the tool chooses a compression ratio from the amount of reduction actually
needed. The calculation includes repeated overlap tokens, preventing overlap
from silently pushing the combined target above the desired context size. Each
pass reports its chosen ratio, chunk ceiling, and overlap. If the result remains
too large, it rechunks and recompresses it (without
source-chunk overlap on later passes) until it fits or reaches the configured
recursive-pass safety limit.

Final-summary length is also adaptive. Very short sources retain close to half
their information, then the target ratio decreases smoothly toward `24%` as the
source approaches 8,000 tokens. Larger sources use the normal ratio, while the
`FINAL_SUMMARY_MAX_OUTPUT_RATIO` cap always leaves generation headroom so the
model can finish rather than being cut off.

Successful Ollama responses are cached atomically. Cache keys include the model,
system prompt, temperature, context size, thinking setting, full request prompt,
and output budget. Consequently, changing the source, prompt, model, or relevant
generation settings produces a new entry instead of reusing stale output. Empty,
corrupt, truncated, and failed responses are not reused. Delete
`.pdf_summarizer_cache` when you want to force a completely fresh run.

When length retries produce several imperfect candidates, the summarizer keeps
the result closest to the requested token range. A slightly oversized but
substantive result therefore wins over a drastically undersized result; equally
distant candidates prefer the shorter version. Truncated generations remain
ineligible regardless of their apparent length.

## Development

Run the test suite with:

```powershell
python -m pytest
```

The tests mock PDF extraction and Ollama calls where needed, so they do not
require a running Ollama server.

## Troubleshooting

- `PDF file not found`: update `PDF_FILE_PATH` or run the command from the
  directory that contains the configured relative path.
- `No extractable text found in PDF`: the PDF may be scanned or image-only. Run
  OCR first and retry with the OCR output PDF.
- `Cannot reach Ollama`: the startup preflight runs after local configuration
  validation but before PDF extraction. Start `ollama serve` or correct
  `--ollama-url`.
- `Ollama model is not available locally`: run the displayed `ollama pull ...`
  command, then retry.
- `Ollama request failed`: transient connection, timeout, rate-limit, and server
  failures are retried automatically. If all attempts fail, confirm `ollama serve`
  is running, `OLLAMA_BASE_URL` is correct, and the configured model has been pulled.
- Context validation errors: lower `CHUNK_LENGTH`, `MAX_CONTEXT_LENGTH`, or
  `LLM_MAX_OUTPUT_TOKENS`, or increase `LLM_CONTEXT_WINDOW` if your model and
  hardware support it.
- Recursive compression did not shrink enough: lower `COMPRESSION_RATIO` or
  `COMPRESSION_MIN_TARGET_RATIO`, increase `MAX_COMPRESSION_RETRIES`, or use a
  model that follows length constraints more reliably.
