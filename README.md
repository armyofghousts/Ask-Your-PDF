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

---

## Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Install Ollama and the model](#install-ollama-and-the-model)
4. [Install the Python packages](#install-the-python-packages)
5. [Add PDF files](#add-pdf-files)
6. [Build the index](#build-the-index)
7. [Start the web application](#start-the-web-application)
8. [Ask a question](#ask-a-question)
9. [How the RAG pipeline works](#how-the-rag-pipeline-works)
10. [Sample questions](#sample-questions)
11. [Example run](#example-run)
12. [Environment variables](#environment-variables)
13. [Sample PDFs](#sample-pdfs)
14. [Project layout](#project-layout)
15. [Limitations](#limitations)

---

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
3. Read the answer. Citations look like `[01_warhammer40k_imperium_and_heresy.pdf, p. 6]`.
4. The sources list shows the retrieved pages, a similarity score, and a short excerpt.

An empty question asks you to enter one. If the index has not been built, the page tells you to run `python ingest.py`. If Ollama is stopped or the model is missing, the page shows that error instead of crashing.

You can ask the same question from PowerShell:

```powershell
python rag.py "What is horus heresy"
```

## How the RAG pipeline works

1. **Ingest.** `ingest.py` opens each PDF and reads each page separately. Page numbers start at 1. Pages with no selectable text are skipped. The page text is split into windows of about 800 characters that overlap by 150 characters.
2. **Embed.** Each chunk is turned into a vector by `all-MiniLM-L6-v2`. Vectors are normalized, and ChromaDB compares them with cosine distance. Cosine similarity is `1 - distance`, so a higher similarity is a closer match.
3. **Store.** Chunks go into `chroma_db`, with the file name, page number, and a hash of the PDF. The same file is skipped on the next run. A changed file replaces its old chunks. A file removed from `pdfs` is removed from the index.
4. **Retrieve.** The question is embedded with the same model. ChromaDB returns the 5 nearest chunks. Chunks below `RELEVANCE_THRESHOLD` are discarded. If none remain, the answer is exactly `I don't know based on the provided documents.`
5. **Generate.** The remaining chunks are labeled `S1`, `S2`, and so on, and sent to Ollama with the question. The model is told to use only that text.
6. **Check citations.** Tags such as `[S1]` are rewritten as `[filename.pdf, p. N]`. A citation is kept only when that file and page were actually retrieved. This verifies the citation reference, not whether the cited passage fully supports every claim in the answer. If the model writes a file or page that was not retrieved, that citation is removed. An answer with no supported citation becomes `I don't know based on the provided documents.` If the model forgets the tag but repeats a long phrase from a retrieved passage, the real page citation is added.

The similarity cutoff is not universal. It was chosen for `all-MiniLM-L6-v2`. On the current PDFs, "What is horus heresy" scored 0.644, "What are the space marines" scored 0.592, and "Who won the 2022 FIFA World Cup?" scored 0.206. The default cutoff is 0.45. Another embedding model needs its own value.

## Sample questions

These answers were produced on this computer on 8 November 2026 with `llama3.2:3b` and the index built from the four PDFs now in `pdfs`. A later run can differ slightly.

### What is horus heresy

```powershell
python rag.py "What is horus heresy"
```

Answer:

> The Horus Heresy is described as the civil war between forces loyal to the Emperor and those aligned with Horus, which transformed the future of humanity (S1). It is also referred to as the name that describes the civil war itself, indicating a conflict that had a profound impact on the Imperium (S1). [01_warhammer40k_imperium_and_heresy.pdf, p. 6]

Page 6 is the chapter "Horus and the road to rebellion". The best similarity was 0.644. The model wrote `(S1)` as well as the checked citation `[01_warhammer40k_imperium_and_heresy.pdf, p. 6]`.

### What are the space marines

```powershell
python rag.py "What are the space marines"
```

Answer:

> According to the passage [02_warhammer40k_factions_and_warp.pdf, p. 3], page 3, Space Marines, also known as Adeptus Astartes, are genetically enhanced human warriors organized into Chapters in the later Imperium.

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
01_warhammer40k_imperium_and_heresy.pdf: unchanged, skipped
02_warhammer40k_factions_and_warp.pdf: unchanged, skipped
03_space_and_astronomy.pdf: unchanged, skipped
04_total_war_warhammer_40000_upcoming_game.pdf: unchanged, skipped

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

The index held 89 chunks: 24 from `01_warhammer40k_imperium_and_heresy.pdf`, 22 from `02_warhammer40k_factions_and_warp.pdf`, 21 from `03_space_and_astronomy.pdf`, and 22 from `04_total_war_warhammer_40000_upcoming_game.pdf`. Each file has 10 pages.

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

## Sample PDFs

The four PDFs now in `pdfs` have selectable text, 10 pages each. None of them prints a source URL or a license, so those cannot be recorded from the files.

| File | Title in the PDF |
| --- | --- |
| `01_warhammer40k_imperium_and_heresy.pdf` | Warhammer 40,000: The Imperium and the Horus Heresy |
| `02_warhammer40k_factions_and_warp.pdf` | Warhammer 40,000: Factions, the Warp, and Galactic Conflict |
| `03_space_and_astronomy.pdf` | Space and Astronomy: A Ten-Chapter Introduction |
| `04_total_war_warhammer_40000_upcoming_game.pdf` | Total War: Warhammer 40,000 — Setting, Strategy, and Expectations |

## Project layout

```text
ask-your-pdfs/
├── pdfs/
├── templates/index.html
├── static/style.css
├── app.py
├── ingest.py
├── rag.py
├── config.py
├── store.py
├── requirements.txt
├── .gitignore
└── README.md
```

`config.py` holds paths and environment settings. `store.py` opens the ChromaDB collection and loads the embedding model once, so ingestion and question answering use the same model.

## Limitations

- Scanned PDFs that do not already contain selectable text are skipped. This project does not run OCR.
- Password-protected PDFs are reported as unreadable and are not indexed.
- A page number in the citation is the PDF page, which can differ from a printed page number inside the document.
- Headers and footers are kept in the text, so some chunks repeat a running header.
- `llama3.2:3b` is a small model. It can phrase an answer awkwardly, as in the Horus Heresy example, which still contains `(S1)` beside the checked page citation. Check the sources list.
- The relevance cutoff is tuned for `all-MiniLM-L6-v2`. Set `RELEVANCE_THRESHOLD` if you change the embedding model, then delete `chroma_db` and run `python ingest.py` again.
- The web page has no accounts, no file upload, and no chat history. Ollama must be running locally (or at the configured server address) to generate answers.
