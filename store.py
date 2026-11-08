import config

_model = None
_PrecomputedEmbeddings = None


def _precomputed_embeddings_class():
    global _PrecomputedEmbeddings
    if _PrecomputedEmbeddings is not None:
        return _PrecomputedEmbeddings

    from chromadb.api.types import Documents, EmbeddingFunction
    from chromadb.utils.embedding_functions import register_embedding_function

    @register_embedding_function
    class PrecomputedEmbeddings(EmbeddingFunction[Documents]):
        def __init__(self):
            pass

        def __call__(self, input):
            raise RuntimeError(
                "Pass embeddings computed by sentence-transformers. "
                "This collection does not embed text on its own."
            )

        @staticmethod
        def name() -> str:
            return "precomputed-sentence-transformers"

        def get_config(self):
            return {}

        @staticmethod
        def build_from_config(config):
            return PrecomputedEmbeddings()

    _PrecomputedEmbeddings = PrecomputedEmbeddings
    return _PrecomputedEmbeddings


def open_collection(chroma_dir=None):
    import chromadb

    path = config.chroma_dir() if chroma_dir is None else chroma_dir
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=_precomputed_embeddings_class()(),
    )
    space = distance_space(collection)
    if space not in (None, "cosine"):
        raise RuntimeError(
            "The vector index is not using cosine distance. "
            "Delete the chroma_db folder and run python ingest.py again."
        )
    return collection


def distance_space(collection):
    metadata = getattr(collection, "metadata", None) or {}
    if isinstance(metadata, dict) and metadata.get("hnsw:space"):
        return metadata.get("hnsw:space")

    configuration = getattr(collection, "configuration", None)
    if configuration is None:
        return None
    if isinstance(configuration, dict):
        hnsw = configuration.get("hnsw") or {}
    else:
        hnsw = getattr(configuration, "hnsw", None)
    if hnsw is None:
        return None
    if isinstance(hnsw, dict):
        space = hnsw.get("space")
    else:
        space = getattr(hnsw, "space", None)
    if hasattr(space, "value"):
        return space.value
    return str(space) if space is not None else None


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = _embedding_model()
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [vector.tolist() for vector in vectors]


def _embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model {config.EMBEDDING_MODEL}...", flush=True)
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model
