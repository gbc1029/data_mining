from functools import cache
from hashlib import sha256
from pathlib import Path
from threading import Lock


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EvidenceIndex:
    def __init__(self, directory):
        import chromadb
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL_NAME, device="cpu")
        self.splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.model.tokenizer, chunk_size=200, chunk_overlap=30
        )
        self.collection = chromadb.PersistentClient(path=str(directory)).get_or_create_collection(
            "minilm_abstracts", embedding_function=None
        )
        self.lock = Lock()

    def search(self, query, papers):
        """Return each candidate's best abstract passage in semantic relevance order."""
        chunks = {}
        for paper in papers:
            for passage in self.splitter.split_text(paper["abstract"]):
                document = f"{paper['title']}\n{passage}"
                key = sha256(f"{paper['id']}\n{document}".encode()).hexdigest()
                chunks[key] = (paper["id"], passage, document)
        if not chunks:
            return {}

        ids = list(chunks)
        with self.lock:
            existing = set(self.collection.get(ids=ids, include=[])["ids"])
            missing = [key for key in ids if key not in existing]
            embeddings = self.model.encode(
                [query] + [chunks[key][2] for key in missing],
                normalize_embeddings=True, show_progress_bar=False,
            ).tolist()
            if missing:
                self.collection.upsert(
                    ids=missing, embeddings=embeddings[1:],
                    documents=[chunks[key][1] for key in missing],
                )
            ranked = self.collection.query(
                query_embeddings=embeddings[:1], ids=ids, n_results=len(ids), include=[]
            )["ids"][0]

        passages = {}
        for key in ranked:
            paper_id, passage, _ = chunks[key]
            passages.setdefault(paper_id, passage)
        return passages


@cache
def get_index():
    return EvidenceIndex(Path(__file__).resolve().parent / "chroma_db")
