"""Retrieve PDF chunks and ask a local Ollama model for a cited answer."""

import re

import requests

import config
from store import embed_texts, open_collection

DONT_KNOW = config.DONT_KNOW

CITATION_RE = re.compile(r"\[([^\[\]\n]+?),\s*p\.\s*(\d+)\]")
SOURCE_TAG_RE = re.compile(r"\[S(\d+)\]", re.IGNORECASE)

SYSTEM_PROMPT = f"""You answer questions about the user's PDF documents.
Use only the source passages in the user message.
If the passages do not contain the answer, reply with exactly this sentence and nothing else:
{DONT_KNOW}
When you use information from a passage, cite its source tag inline, exactly like [S1].
You may cite more than one tag. Use only tags that appear in the user message.
Do not invent facts, filenames, page numbers, or source tags.
Write a short answer in plain sentences."""


class IndexEmptyError(Exception):
    """Raised when ChromaDB has no chunks yet."""


class OllamaError(Exception):
    """Raised when the local Ollama server cannot answer."""


def cosine_similarity(distance) -> float:
    """Convert Chroma cosine distance into a similarity score.

    For a collection created with hnsw:space cosine, distance = 1 - cosine
    similarity. This conversion is not meaningful for L2 distance.
    """
    return 1.0 - float(distance)


def citation_for(filename, page) -> str:
    return f"[{filename}, p. {int(page)}]"


def format_citations(answer, sources) -> str:
    """Rewrite source tags and drop citations that were not retrieved.

    A citation is kept only when its filename and page were in the retrieved
    chunks. Answers that end up with no valid citation become the standard
    "I don't know" sentence, so the model cannot invent a source.
    """
    by_tag = {}
    allowed = set()
    for source in sources:
        by_tag[str(source["tag"]).upper()] = source
        allowed.add((source["filename"], int(source["page"])))

    def replace_tag(match):
        source = by_tag.get(f"S{int(match.group(1))}")
        if source is None:
            return ""
        return citation_for(source["filename"], source["page"])

    text = SOURCE_TAG_RE.sub(replace_tag, answer or "")

    def replace_explicit(match):
        filename = match.group(1).strip()
        page = int(match.group(2))
        if (filename, page) not in allowed:
            return ""
        return citation_for(filename, page)

    text = CITATION_RE.sub(replace_explicit, text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r" +([.,;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = _collapse_repeated_citations(text)

    if not text or DONT_KNOW in text:
        return DONT_KNOW
    if CITATION_RE.search(text):
        return text
    if _is_refusal(text):
        return DONT_KNOW

    # The model sometimes answers from a passage without emitting a source tag.
    # Keep the answer only when a long phrase from it appears in a retrieved
    # passage, and attach that passage's real filename and page.
    supported = _overlapping_sources(text, sources)
    if not supported:
        return DONT_KNOW
    citations = []
    for source in supported:
        citation = citation_for(source["filename"], source["page"])
        if citation not in citations:
            citations.append(citation)
    return text.rstrip() + " " + " ".join(citations)


def _is_refusal(text) -> bool:
    lowered = text.lower()
    phrases = (
        "don't know",
        "do not know",
        "dont know",
        "cannot answer",
        "can't answer",
        "not enough information",
    )
    return any(phrase in lowered for phrase in phrases)


def _collapse_repeated_citations(text) -> str:
    return re.sub(
        r"(\[[^\[\]\n]+, p\. \d+\])(?:\s*(?:,|and)\s*\1)+",
        r"\1",
        text,
    )


def _overlapping_sources(answer, sources):
    """Return sources that contain a long phrase from the answer."""
    words = " ".join(answer.lower().split()).split()
    if len(words) < 6:
        return []
    matched = []
    for source in sources:
        source_text = " ".join(source["text"].lower().split())
        for start in range(len(words) - 5):
            phrase = " ".join(words[start : start + 8])
            if phrase in source_text:
                matched.append(source)
                break
    return matched


def retrieve(question, chroma_dir=None, embedder=None):
    """Return the nearest chunks, best match first.

    Each hit includes filename, 1-based page, text, distance, and similarity.
    The caller decides which hits clear the relevance threshold.
    """
    collection = open_collection(chroma_dir)
    if collection.count() == 0:
        raise IndexEmptyError(
            "The document index is empty. Add PDF files to the pdfs folder and run python ingest.py."
        )

    vectors = embedder([question]) if embedder else embed_texts([question])
    if not vectors:
        raise RuntimeError("Could not embed the question.")

    result_count = min(config.TOP_K, collection.count())
    found = collection.query(
        query_embeddings=[vectors[0]],
        n_results=result_count,
        include=["documents", "metadatas", "distances"],
    )

    documents = (found.get("documents") or [[]])[0] or []
    metadatas = (found.get("metadatas") or [[]])[0] or []
    distances = (found.get("distances") or [[]])[0] or []

    hits = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        if not document or not metadata:
            continue
        hits.append(
            {
                "filename": metadata.get("filename", ""),
                "page": int(metadata.get("page", 0)),
                "text": document,
                "distance": float(distance),
                "similarity": cosine_similarity(distance),
            }
        )
    return hits


def build_messages(question, sources):
    blocks = []
    for source in sources:
        blocks.append(
            "\n".join(
                [
                    f"[{source['tag']}]",
                    f"filename: {source['filename']}",
                    f"page: {source['page']}",
                    "passage:",
                    source["text"],
                ]
            )
        )
    user = (
        "Sources:\n\n"
        + "\n\n".join(blocks)
        + "\n\nQuestion: "
        + question.strip()
        + "\n\nCite the passages you use with their tags, such as [S1]. "
        + f"If the passages do not contain the answer, reply with exactly: {DONT_KNOW}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def ensure_ollama():
    """Fail with a short message if Ollama or the configured model is missing."""
    base_url = config.ollama_base_url()
    model = config.ollama_model()
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {base_url}. "
            "Start Ollama and try again."
        ) from exc
    except requests.Timeout as exc:
        raise OllamaError(
            f"Timed out connecting to Ollama at {base_url}."
        ) from exc
    except requests.RequestException as exc:
        raise OllamaError(
            f"Could not reach Ollama at {base_url}: {exc}"
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise OllamaError(
            f"Ollama at {base_url} returned an unexpected response."
        ) from exc

    names = [item.get("name", "") for item in payload.get("models", [])]
    if model not in names:
        raise OllamaError(
            f"The Ollama model '{model}' is not installed. Run: ollama pull {model}"
        )


def generate_answer(question, sources):
    """Send the question and the retrieved passages to Ollama."""
    ensure_ollama()
    base_url = config.ollama_base_url()
    model = config.ollama_model()
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": build_messages(question, sources),
                "stream": False,
                "options": {"temperature": 0},
            },
            timeout=config.ollama_timeout(),
        )
    except requests.ConnectionError as exc:
        raise OllamaError(
            f"Could not connect to Ollama at {base_url}. Start Ollama and try again."
        ) from exc
    except requests.Timeout as exc:
        raise OllamaError(
            "Ollama took too long to respond. Wait a moment and try again, "
            "or increase OLLAMA_TIMEOUT."
        ) from exc
    except requests.RequestException as exc:
        raise OllamaError(f"The request to Ollama failed: {exc}") from exc

    if response.status_code == 404:
        raise OllamaError(
            f"The Ollama model '{model}' is not installed. Run: ollama pull {model}"
        )
    if not response.ok:
        detail = _error_detail(response)
        raise OllamaError(
            f"Ollama returned status {response.status_code}. {detail}".strip()
        )

    try:
        content = response.json().get("message", {}).get("content", "")
    except ValueError as exc:
        raise OllamaError("Ollama returned a response that was not valid JSON.") from exc
    content = (content or "").strip()
    if not content:
        raise OllamaError("Ollama returned an empty response.")
    return content


def _error_detail(response) -> str:
    try:
        return str(response.json().get("error", "")).strip()
    except ValueError:
        return response.text[:200].strip()


def _snippet(text, limit=280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _public_sources(sources):
    published = []
    for source in sources:
        published.append(
            {
                "filename": source["filename"],
                "page": int(source["page"]),
                "citation": citation_for(source["filename"], source["page"]),
                "snippet": _snippet(source["text"]),
                "similarity": round(float(source["similarity"]), 3),
            }
        )
    return published


def answer_question(question, chroma_dir=None, embedder=None):
    """Answer one question. Returns answer, sources, and an error message."""
    cleaned = (question or "").strip()
    if not cleaned:
        return _result(error="Please enter a question.")

    try:
        threshold = config.relevance_threshold()
    except RuntimeError as exc:
        return _result(error=str(exc))

    try:
        hits = retrieve(cleaned, chroma_dir=chroma_dir, embedder=embedder)
    except IndexEmptyError as exc:
        return _result(error=str(exc))
    except Exception as exc:
        return _result(
            error=(
                "Could not read the document index. Delete the chroma_db folder "
                f"and run python ingest.py again. ({exc})"
            )
        )

    relevant = [hit for hit in hits if hit["similarity"] >= threshold]
    if not relevant:
        return _result(answer=DONT_KNOW)

    for index, hit in enumerate(relevant, start=1):
        hit["tag"] = f"S{index}"

    try:
        raw_answer = generate_answer(cleaned, relevant)
    except OllamaError as exc:
        return _result(error=str(exc))
    except Exception as exc:
        return _result(error=f"Could not generate an answer: {exc}")

    answer = format_citations(raw_answer, relevant)
    sources = _public_sources(relevant) if answer != DONT_KNOW else []
    return _result(answer=answer, sources=sources)


def _result(answer=None, sources=None, error=None):
    return {
        "answer": answer,
        "sources": sources or [],
        "error": error,
    }


def main():
    import sys

    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('Usage: python rag.py "your question"')
        return 1
    result = answer_question(question)
    if result["error"]:
        print(result["error"])
        return 1
    print(result["answer"])
    if result["sources"]:
        print()
        print("Sources:")
        for source in result["sources"]:
            print(f"{source['citation']} (similarity {source['similarity']:.2f})")
            print(source["snippet"])
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
