"""LanceDB vector store for semantic memory search. Lazy-loaded."""

import importlib
import logging
import os

log = logging.getLogger(__name__)

_LANCEDB_AVAILABLE = importlib.util.find_spec("lancedb") is not None
_ST_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None


def _sanitize_str(value: str) -> str:
    """Strip characters that could break SQL-like WHERE clauses."""
    return value.replace("'", "").replace("\\", "").replace(";", "")


class MemoryVectorStore:

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    VECTOR_DIM = 384

    def __init__(self, db_path: str, model_name: str = None):
        self._db_path = db_path
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None  # lazy loaded
        self._db = None
        self._table = None

        if self.enabled:
            os.makedirs(db_path, exist_ok=True)

    @property
    def enabled(self) -> bool:
        """True if both lancedb and sentence-transformers are available."""
        return _LANCEDB_AVAILABLE and _ST_AVAILABLE

    def _ensure_model(self):
        if self._model is not None:
            return
        if not _ST_AVAILABLE:
            raise RuntimeError("sentence-transformers not installed")
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name, device="cpu")
        log.info("Embedding model loaded (%d dim)", self.VECTOR_DIM)

    def _ensure_table(self):
        if self._table is not None:
            return
        if not _LANCEDB_AVAILABLE:
            raise RuntimeError("lancedb not installed")

        import lancedb
        self._db = lancedb.connect(self._db_path)

        if "memories" in self._db.table_names():
            self._table = self._db.open_table("memories")
        else:
            import pyarrow as pa
            schema = pa.schema([
                pa.field("id", pa.int64()),
                pa.field("user_id", pa.utf8()),
                pa.field("text", pa.utf8()),
                pa.field("vector", pa.list_(pa.float32(), self.VECTOR_DIM)),
            ])
            self._table = self._db.create_table("memories", schema=schema)
            log.info("Created LanceDB memories table")

    def _embed(self, text: str) -> list[float]:
        self._ensure_model()
        return self._model.encode(text).tolist()

    def add(self, memory_id: int, user_id: str, text: str):
        """Upsert a memory into the vector index."""
        if not self.enabled:
            return
        self._ensure_table()
        log.debug("VectorDB add: id=%d user=%s text='%s'", memory_id, user_id, text[:80])
        vector = self._embed(text)
        try:
            self._table.delete(f"id = {int(memory_id)}")
        except Exception:
            pass
        self._table.add([{
            "id": memory_id,
            "user_id": user_id,
            "text": text,
            "vector": vector,
        }])

    def search(self, query: str, user_id: str = None, limit: int = 5) -> list[dict]:
        if not self.enabled:
            return []
        self._ensure_table()

        row_count = self._table.count_rows()
        log.debug("VectorDB search: query='%s', user=%s, limit=%d, rows=%d",
                  query[:80], user_id, limit, row_count)

        if row_count == 0:
            return []

        vector = self._embed(query)
        results = self._table.search(vector).limit(limit)

        if user_id:
            results = results.where(f"user_id = '{_sanitize_str(user_id)}'")

        try:
            df = results.to_pandas()
        except Exception:
            return []

        memories = []
        for _, row in df.iterrows():
            memories.append({
                "id": int(row["id"]),
                "user_id": row["user_id"],
                "text": row["text"],
                "score": float(row.get("_distance", 0)),
            })
        log.debug("VectorDB results: %d hits", len(memories))
        return memories

    def delete(self, memory_id: int):
        if not self.enabled:
            return
        self._ensure_table()
        try:
            self._table.delete(f"id = {int(memory_id)}")
        except Exception:
            pass

    def delete_all_for_user(self, user_id: str):
        if not self.enabled:
            return
        self._ensure_table()
        try:
            self._table.delete(f"user_id = '{_sanitize_str(user_id)}'")
            log.info("Deleted vector memories for user %s", user_id)
        except Exception as e:
            log.warning("Failed to delete vector memories for %s: %s", user_id, e)
