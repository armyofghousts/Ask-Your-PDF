# Ask Your PDFs

A small local RAG app. It reads PDFs from a folder, finds the passages that match a question, and asks a local language model to answer from those passages only.

Answers cite the file and page as `[filename.pdf, p. N]`. No paid API and no API key are required. The language model runs through [Ollama](https://ollama.com/) on your own computer.

---

## Quick start

```powershell
cd path\to\ask-your-pdfs
pip install -r requirements.txt
ollama pull llama3.2:3b
python ingest.py
python app.py
```

Then open <http://127.0.0.1:5000>.

Ollama is a separate application. It is not installed by `pip install -r requirements.txt`.

## Features

- Extracts text from each PDF page with PyMuPDF.
- Splits pages into chunks of about 800 characters, with 150 characters of overlap.
- Embeds chunks with `sentence-transformers` (`all-MiniLM-L6-v2`).
- Stores chunks in a persistent ChromaDB index, including the file name and 1-based page number.
- Skips unchanged PDFs on a later run, replaces chunks when a PDF changes, and drops chunks when a PDF is removed.
- Retrieves the 5 nearest chunks and keeps only those above a configurable cosine-similarity threshold.
- Asks Ollama to answer from those chunks, then checks citations against the retrieved pages.
- Shows the answer, the citations, and a short source excerpt in a single web page.

## Requirements

- Windows with PowerShell
- Python 3.10 or newer (tested with Python 3.13.17)
- Ollama, for generating answers
- An internet connection the first time, so the embedding model can be downloaded

A virtual environment is optional. The commands below install into the Python you have on `PATH`.

## Install Ollama and the model

Ollama is a separate application. It is not installed by `pip install -r requirements.txt` and must be installed separately before you can generate answers.

1. Install Ollama from <https://ollama.com/download>, or in PowerShell:

   ```powershell
   winget install --id Ollama.Ollama -e --source winget --accept-package-agreements --accept-source-agreements
   ```

2. Open a new PowerShell window so `ollama` is on `PATH`. The Ollama application also starts a local server at `http://localhost:11434`. If the Ollama service is not running, start **Ollama** from the Start menu. The API is available at `http://localhost:11434`.

3. Download the model used by this project:

   ```powershell
   ollama pull llama3.2:3b
   ```

4. Check that the model is present:

   ```powershell
   ollama list
   ```

You should see `llama3.2:3b`. This model is about 2 GB.

### If Ollama does not connect

| What you see | What to do |
| --- | --- |
| `Could not connect to Ollama at http://localhost:11434` | The Ollama application is not running. Start it from the Start menu. |
| `The Ollama model 'llama3.2:3b' is not installed` | The pull step has not finished. Run `ollama pull llama3.2:3b`. |
| PowerShell says `ollama` is not recognized | Close the terminal and open a new one, or call the program directly. |

```powershell
& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list
```

## Install the Python packages

Open PowerShell and navigate to the folder where you downloaded or extracted the project (replace `path\to` with your actual path):

```powershell
cd path\to\ask-your-pdfs
pip install -r requirements.txt
```

If pip prints `Defaulting to user installation`, that is fine. The first ingestion also downloads `all-MiniLM-L6-v2` from Hugging Face (about 90 MB). No Hugging Face account is required. A warning about unauthenticated requests can be ignored.

Optional virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Add PDF files

Put `.pdf` files in the `pdfs` folder. Four sample documents are already there. The application does not upload files in the browser. After adding, removing, or replacing a PDF, run ingestion again.

## Build the index

```powershell
python ingest.py
```

The command reads every PDF, stores new or changed chunks, and prints a summary. Running it again does not duplicate chunks for files that have not changed.

## Start the web application

```powershell
python app.py
```

Open <http://127.0.0.1:5000>.

## Ask a question

1. Type a question into the box.
2. Click **Ask**.
3. Read the answer. Citations look like `[warhammer40k_imperium_and_heresy.pdf, p. 6]`.
4. The sources list shows the retrieved pages, a similarity score, and a short excerpt.

An empty question asks you to enter one. If the index has not been built, the page tells you to run `python ingest.py`. If Ollama is stopped or the model is missing, the page shows that error instead of crashing.

You can ask the same question from PowerShell:

```powershell
python rag.py "What is horus heresy"
```

## Sample questions

These answers were produced on this computer on 8 November 2026 with `llama3.2:3b` and the index built from the four PDFs now in `pdfs`. A later run can differ slightly.

### What is horus heresy

```powershell
python rag.py "What is horus heresy"
```

Answer:

> The Horus Heresy is described as the civil war between forces loyal to the Emperor and those aligned with Horus, which transformed the future of humanity (S1). It is also referred to as the name that describes the civil war itself, indicating a conflict that had a profound impact on the Imperium (S1). [warhammer40k_imperium_and_heresy.pdf, p. 6]

Page 6 is the chapter "Horus and the road to rebellion". The best similarity was 0.644. The model wrote `(S1)` as well as the checked citation `[warhammer40k_imperium_and_heresy.pdf, p. 6]`.

### What are the space marines

```powershell
python rag.py "What are the space marines"
```

Answer:

> According to the passage [warhammer40k_factions_and_warp.pdf, p. 3], page 3, Space Marines, also known as Adeptus Astartes, are genetically enhanced human warriors organized into Chapters in the later Imperium.

That page says Space Marines, or Adeptus Astartes, are genetically enhanced human warriors organized into Chapters. The best similarity was 0.592.

### A question the documents cannot answer

```powershell
python rag.py "Who won the 2022 FIFA World Cup?"
```

The run returned exactly:

```text
I don't know based on the provided documents.
```

The best retrieved similarity was 0.206, which is below the 0.45 cutoff, so Ollama was not asked to guess.

## Example run

`python ingest.py` with the current PDFs already in the index:

```text
warhammer40k_imperium_and_heresy.pdf: unchanged, skipped
warhammer40k_factions_and_warp.pdf: unchanged, skipped
space_and_astronomy.pdf: unchanged, skipped
total_war_warhammer_40000_upcoming_game.pdf: unchanged, skipped

Ingestion summary
PDFs found: 4
PDFs processed: 0
PDFs unchanged: 4
PDFs unreadable: 0
PDFs with no text: 0
Pages seen: 40
Pages with text: 40
Pages without text: 0
Chunks written: 0
Chunks removed: 0
```

The index held 89 chunks: 24 from `warhammer40k_imperium_and_heresy.pdf`, 22 from `warhammer40k_factions_and_warp.pdf`, 21 from `space_and_astronomy.pdf`, and 22 from `total_war_warhammer_40000_upcoming_game.pdf`. Each file has 10 pages.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_MODEL` | `llama3.2:3b` | Model name sent to Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_TIMEOUT` | `120` | Seconds to wait for an answer |
| `RELEVANCE_THRESHOLD` | `0.45` | Minimum cosine similarity. Higher is stricter. This value is for `all-MiniLM-L6-v2`, not for every model. |
| `PDF_DIR` | `pdfs` next to the code | Folder of PDF files |
| `CHROMA_DIR` | `chroma_db` next to the code | Persistent vector index |

PowerShell example:

```powershell
$env:OLLAMA_MODEL = "llama3.2:3b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:RELEVANCE_THRESHOLD = "0.45"
python app.py
```

`config.py` holds paths and environment settings. `store.py` opens the ChromaDB collection and loads the embedding model once, so ingestion and question answering use the same model.
