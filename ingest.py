"""
MS MARCO → Redis Vector Ingestion
Uses RedisVL HFTextVectorizer with embedding cache.
"""

import os
import time
from typing import Iterator

from datasets import load_dataset
from redisvl.index import SearchIndex
from redisvl.redis.utils import array_to_buffer
from redisvl.utils.vectorize import HFTextVectorizer
from redisvl.extensions.cache.embeddings import EmbeddingsCache

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Config
REDIS_URL = os.getenv("REDIS_URL", "redis://:@localhost:6379")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 500
MAX_PASSAGES = 100  # Set to None for full dataset

# Initialize vectorizer with embedding cache
vectorizer = HFTextVectorizer(
    model=MODEL_NAME,
    # cache=EmbeddingsCache(
    #     name="msmarco_embedcache",
    #     ttl=3600,
    #     redis_url=REDIS_URL,
    # )
)

# Schema for RedisVL
SCHEMA = {
    "index": {
        "name": "msmarco",
        "prefix": "doc",
    },
    "fields": [
        {
            "name": "passage_id",
            "type": "tag",
            "attrs": {"sortable": True},
        },
        {
            "name": "text",
            "type": "text",
        },
        {
            "name": "url",
            "type": "tag",
            "attrs": {"sortable": True},
        },
        {
            "name": "text_embedding",
            "type": "vector",
            "attrs": {
                "algorithm": "hnsw",
                "dims": vectorizer.dims,
                "distance_metric": "cosine",
                "datatype": "float32",
            },
        },
    ],
}


def flatten_passages(dataset, max_passages: int | None = None) -> Iterator[dict]:
    """Flatten MS MARCO passages structure into individual docs."""
    count = 0
    for row in dataset:
        passages = row["passages"]
        for i, (text, url) in enumerate(
            zip(passages["passage_text"], passages["url"])
        ):
            if max_passages and count >= max_passages:
                return
            yield {
                "passage_id": f"{row['query_id']}_{i}",
                "text": text,
                "url": url,
            }
            count += 1


def batch_iterator(iterator: Iterator, batch_size: int) -> Iterator[list]:
    """Yield batches from an iterator."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    print("Loading MS MARCO dataset...")
    ds = load_dataset("ms_marco", "v1.1", split="train", streaming=True)

    print("Creating Redis index...")
    index = SearchIndex.from_dict(SCHEMA, redis_url=REDIS_URL)
    index.create(overwrite=True, drop=True)

    print(f"Starting ingestion (batch_size={BATCH_SIZE})...")
    start = time.time()
    total_docs = 0

    # Process in batches
    passages = flatten_passages(ds, MAX_PASSAGES)
    
    for batch in batch_iterator(passages, BATCH_SIZE):
        texts = [doc["text"] for doc in batch]
        
        # Embed batch using HFTextVectorizer (with caching)
        embeddings = vectorizer.embed_many(texts)
        
        # Prepare data for RedisVL
        data = [
            {
                "passage_id": doc["passage_id"],
                "text": doc["text"],
                "url": doc["url"],
                "text_embedding": array_to_buffer(embeddings[i], dtype="float32"),
            }
            for i, doc in enumerate(batch)
        ]
        
        # Load batch into Redis
        index.load(data, id_field="passage_id")
        total_docs += len(batch)
        print(f"  Ingested {total_docs} documents...")

    elapsed = time.time() - start
    print(f"\nDone! Ingested {total_docs} documents in {elapsed:.1f}s")
    print(f"Rate: {total_docs / elapsed:.0f} docs/sec")
    print(f"\nIndex info:")
    print(index.info())


if __name__ == "__main__":
    main()