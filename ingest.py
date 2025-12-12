"""
MS MARCO v2.1 → Redis Vector Ingestion

Supports two embedding modes:
  --use-cohere    Use pre-computed Cohere embed-english-v3 (1024-dim, faster)
  --use-hf        Use local HuggingFace model (384-dim default, flexible)

Usage:
    uv run python ingest.py --use-cohere
    uv run python ingest.py --use-hf
    uv run python ingest.py --use-hf --hf-model sentence-transformers/all-mpnet-base-v2
"""

import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from datasets import load_dataset
from redisvl.index import SearchIndex
from redisvl.redis.utils import array_to_buffer
from redisvl.schema import IndexSchema
from tqdm import tqdm
import redis

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class IngestConfig:
    """Configuration for ingestion."""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    
    index_name: str = "msmarco"
    index_prefix: str = "doc"
    
    # Embedding config
    use_cohere: bool = True
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_dims: int = 1024  # Will be updated based on embedding choice
    
    batch_size: int = 500
    encode_batch_size: int = 128
    max_passages: Optional[int] = 100_000

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"


def create_redis_client(config: IngestConfig) -> redis.Redis:
    """Create a new Redis client."""
    return redis.Redis(
        host=config.redis_host,
        port=config.redis_port,
        password=config.redis_password,
    )


def create_index_schema(config: IngestConfig) -> IndexSchema:
    """Create RedisVL index schema."""
    schema_dict = {
        "index": {
            "name": config.index_name,
            "prefix": config.index_prefix,
            "storage_type": "hash",
        },
        "fields": [
            {
                "name": "docid",
                "type": "tag",
                "attrs": {"sortable": True},
            },
            {
                "name": "title",
                "type": "text",
            },
            {
                "name": "segment",
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
                    "dims": config.vector_dims,
                    "distance_metric": "cosine",
                    "datatype": "float32",
                },
            },
        ],
    }
    return IndexSchema.from_dict(schema_dict)


def get_hf_model_dims(model_name: str) -> int:
    """Get embedding dimensions for a HuggingFace model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return model.get_sentence_embedding_dimension()


def main():
    parser = argparse.ArgumentParser(
        description="Ingest MS MARCO v2.1 to Redis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Embedding Options:
  --use-cohere    Use pre-computed Cohere embed-english-v3 embeddings (1024-dim)
                  Faster ingestion, no GPU needed
  
  --use-hf        Use local HuggingFace SentenceTransformer model
                  Flexible model choice, requires compute

Examples:
  %(prog)s --use-cohere                           # Fast: pre-computed embeddings
  %(prog)s --use-hf                               # Local: all-MiniLM-L6-v2 (384-dim)
  %(prog)s --use-hf --hf-model all-mpnet-base-v2  # Local: mpnet (768-dim)
        """,
    )
    
    # Embedding choice (mutually exclusive)
    emb_group = parser.add_mutually_exclusive_group(required=True)
    emb_group.add_argument(
        "--use-cohere",
        action="store_true",
        help="Use pre-computed Cohere embeddings (1024-dim)",
    )
    emb_group.add_argument(
        "--use-hf",
        action="store_true",
        help="Use local HuggingFace model for embeddings",
    )
    
    parser.add_argument(
        "--hf-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace model to use (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--max-passages",
        type=int,
        default=100_000,
        help="Maximum passages to ingest (default: 100000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for Redis writes (default: 500)",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=128,
        help="Batch size for HF encoding (default: 128)",
    )
    args = parser.parse_args()
    
    # Determine vector dimensions
    if args.use_cohere:
        vector_dims = 1024
    else:
        print(f"🔍 Loading HuggingFace model to get dimensions...")
        vector_dims = get_hf_model_dims(args.hf_model)
        print(f"   Model: {args.hf_model} ({vector_dims}-dim)")
    
    config = IngestConfig(
        use_cohere=args.use_cohere,
        hf_model=args.hf_model,
        vector_dims=vector_dims,
        max_passages=args.max_passages,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
    )
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"   Embedding:      {'Cohere (pre-computed)' if config.use_cohere else f'HuggingFace ({config.hf_model})'}")
    print(f"   Vector dims:    {config.vector_dims}")
    print(f"   Max passages:   {config.max_passages:,}")
    print(f"   Batch size:     {config.batch_size}")
    print(f"{'='*60}\n")
    
    total_start = time.perf_counter()
    
    # 1. Load dataset
    print(f"📂 Loading MS MARCO v2.1...")
    
    if config.use_cohere:
        # Load with pre-computed embeddings
        ds = load_dataset(
            "Cohere/msmarco-v2.1-embed-english-v3",
            "passages",
            data_files={"train": "passages_parquet/msmarco_v2.1_doc_segmented_00.parquet"},
            split="train",
        ).select_columns(["docid", "title", "segment", "url", "emb"])
    else:
        # Load without embeddings (we'll compute them)
        ds = load_dataset(
            "Cohere/msmarco-v2.1-embed-english-v3",
            "passages",
            data_files={"train": "passages_parquet/msmarco_v2.1_doc_segmented_00.parquet"},
            split="train",
        ).select_columns(["docid", "title", "segment", "url"])
    
    # Limit rows if needed
    if config.max_passages and len(ds) > config.max_passages:
        ds = ds.select(range(config.max_passages))
    
    print(f"   ✅ Loaded {len(ds):,} passages")
    
    # 2. Create Redis index
    print(f"\n🗄️  Creating Redis index...")
    redis_client = create_redis_client(config)
    schema = create_index_schema(config)
    index = SearchIndex(schema, redis_client)
    index.create(overwrite=True, drop=True)
    print(f"   ✅ Created index: {config.index_name}")
    
    # 3. Compute embeddings if using HuggingFace
    embeddings = None
    embed_elapsed = 0
    
    if not config.use_cohere:
        from sentence_transformers import SentenceTransformer
        
        print(f"\n🔢 Computing embeddings with {config.hf_model}...")
        model = SentenceTransformer(config.hf_model)
        
        texts = [row["segment"] for row in ds]
        
        embed_start = time.perf_counter()
        embeddings = model.encode(
            texts,
            batch_size=config.encode_batch_size,
            show_progress_bar=True,
        )
        embed_elapsed = time.perf_counter() - embed_start
        
        print(f"   ✅ Embedded in {embed_elapsed:.1f}s ({len(texts) / embed_elapsed:.0f} passages/sec)")
    
    # 4. Write to Redis
    print(f"\n📥 Writing to Redis (batch_size={config.batch_size})...")
    
    write_start = time.perf_counter()
    total_docs = 0
    
    for i in tqdm(range(0, len(ds), config.batch_size), desc="Writing"):
        batch_end = min(i + config.batch_size, len(ds))
        batch = ds.select(range(i, batch_end))
        
        with redis_client.pipeline(transaction=False) as pipe:
            for j, row in enumerate(batch):
                key = f"{config.index_prefix}:{row['docid']}"
                
                # Get embedding from pre-computed or locally computed
                if config.use_cohere:
                    emb = np.array(row["emb"], dtype="float32")
                else:
                    emb = embeddings[i + j].astype("float32")
                
                pipe.hset(key, mapping={
                    "docid": row["docid"],
                    "title": row["title"],
                    "segment": row["segment"],
                    "url": row["url"],
                    "text_embedding": array_to_buffer(emb, dtype="float32"),
                })
            pipe.execute()
        
        total_docs += len(batch)
    
    write_elapsed = time.perf_counter() - write_start
    total_elapsed = time.perf_counter() - total_start
    
    # 5. Summary
    print(f"\n{'='*60}")
    print(f"✅ DONE!")
    print(f"   Total documents: {total_docs:,}")
    if embed_elapsed > 0:
        print(f"   Embed time:      {embed_elapsed:.1f}s ({total_docs / embed_elapsed:.0f}/sec)")
    print(f"   Write time:      {write_elapsed:.1f}s ({total_docs / write_elapsed:.0f}/sec)")
    print(f"   Total time:      {total_elapsed:.1f}s")
    print(f"   Overall rate:    {total_docs / total_elapsed:.0f} docs/sec")
    print(f"{'='*60}")
    
    # Show index info
    print(f"\n📊 Index info:")
    info = index.info()
    print(f"   num_docs: {info.get('num_docs', 'N/A')}")


if __name__ == "__main__":
    main()