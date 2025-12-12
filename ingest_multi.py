"""
MS MARCO v2.1 → Redis Vector Ingestion (Multiprocessing)

Supports two embedding modes:
  --use-cohere    Use pre-computed Cohere embed-english-v3 (1024-dim, faster)
  --use-hf        Use local HuggingFace model (384-dim default, flexible)

Usage:
    uv run python ingest_multi.py --use-cohere --workers 8
    uv run python ingest_multi.py --use-hf --embed-workers 1 --redis-workers 8
"""

import argparse
import os
import time
from dataclasses import dataclass
from multiprocessing import cpu_count, get_context
from typing import List, Optional, Tuple

import numpy as np
import torch
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
    vector_dims: int = 1024
    
    batch_size: int = 500
    encode_batch_size: int = 128
    max_passages: Optional[int] = 100_000
    
    embed_workers: int = 1
    redis_workers: int = 1
    mp_start_method: str = "spawn"

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


# =============================================================================
# Device detection for HuggingFace embeddings
# =============================================================================

def get_target_devices(embed_workers: int) -> Tuple[List[str], str]:
    """Get target devices for embedding multiprocessing."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if embed_workers > num_gpus:
            print(f"   ⚠️  Requested {embed_workers} embed workers but only {num_gpus} GPUs available")
        devices = [f"cuda:{i}" for i in range(min(embed_workers, num_gpus))]
        return devices, f"CUDA GPU(s): {devices}"
    elif torch.backends.mps.is_available():
        if embed_workers > 1:
            print(f"   ⚠️  MPS only supports 1 embed worker (requested {embed_workers})")
        return ["mps"], "MPS (Apple Silicon)"
    else:
        devices = ["cpu"] * embed_workers
        return devices, f"{embed_workers} CPU worker(s)"


def get_default_embed_workers() -> int:
    """Get sensible default for embedding workers."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    elif torch.backends.mps.is_available():
        return 1
    else:
        return max(1, cpu_count() - 1)


# =============================================================================
# Multiprocessing workers for Redis writes
# =============================================================================

_redis_client: Optional[redis.Redis] = None
_config: Optional[IngestConfig] = None


def init_redis_worker(config_dict: dict):
    """Initialize Redis client in each worker process."""
    global _redis_client, _config
    _config = IngestConfig(**config_dict)
    _redis_client = create_redis_client(_config)


def write_batch_worker(batch: List[dict]) -> Tuple[int, float]:
    """Write a batch of documents to Redis."""
    global _redis_client, _config
    
    start = time.perf_counter()
    
    with _redis_client.pipeline(transaction=False) as pipe:
        for row in batch:
            key = f"{_config.index_prefix}:{row['docid']}"
            pipe.hset(key, mapping={
                "docid": row["docid"],
                "title": row["title"],
                "segment": row["segment"],
                "url": row["url"],
                "text_embedding": array_to_buffer(
                    np.array(row["emb"], dtype="float32"),
                    dtype="float32"
                ),
            })
        pipe.execute()
    
    elapsed_ms = (time.perf_counter() - start) * 1000
    return len(batch), elapsed_ms


def batch_list(items: List, batch_size: int) -> List[List]:
    """Split a list into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


# =============================================================================
# Embedding functions
# =============================================================================

def embed_with_hf(
    texts: List[str],
    model_name: str,
    embed_workers: int,
    encode_batch_size: int,
) -> np.ndarray:
    """Embed texts using HuggingFace SentenceTransformers."""
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    target_devices, device_desc = get_target_devices(embed_workers)
    print(f"   Devices: {device_desc}")
    
    if len(target_devices) == 1:
        # Single device
        if target_devices[0] != "cpu":
            model = model.to(target_devices[0])
        
        embeddings = model.encode(
            texts,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            device=target_devices[0],
        )
    else:
        # Multi-device using native SentenceTransformers pool
        pool = model.start_multi_process_pool(target_devices=target_devices)
        try:
            embeddings = model.encode_multi_process(
                texts,
                pool,
                batch_size=encode_batch_size,
            )
        finally:
            model.stop_multi_process_pool(pool)
    
    return embeddings


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ingest MS MARCO v2.1 to Redis (multiprocessing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Embedding Options:
  --use-cohere    Use pre-computed Cohere embeddings (1024-dim, fast)
  --use-hf        Use local HuggingFace model (flexible, requires compute)

Examples:
  %(prog)s --use-cohere --workers 8
  %(prog)s --use-hf --embed-workers 1 --redis-workers 8
  %(prog)s --use-hf --hf-model all-mpnet-base-v2 --embed-workers 4
        """,
    )
    
    # Embedding choice
    emb_group = parser.add_mutually_exclusive_group(required=True)
    emb_group.add_argument("--use-cohere", action="store_true", help="Use pre-computed Cohere embeddings")
    emb_group.add_argument("--use-hf", action="store_true", help="Use local HuggingFace model")
    
    parser.add_argument("--hf-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace model (default: all-MiniLM-L6-v2)")
    
    # Worker counts
    parser.add_argument("--workers", type=int, default=None,
                        help="Workers for Redis writes (shorthand, default: CPU count - 1)")
    parser.add_argument("--redis-workers", type=int, default=None,
                        help="Workers for Redis writes (default: CPU count - 1)")
    parser.add_argument("--embed-workers", type=int, default=None,
                        help="Workers for HF embedding (default: auto-detect)")
    
    parser.add_argument("--max-passages", type=int, default=100_000,
                        help="Maximum passages to ingest (default: 100000)")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Batch size for Redis writes (default: 500)")
    parser.add_argument("--encode-batch-size", type=int, default=128,
                        help="Batch size for HF encoding (default: 128)")
    
    args = parser.parse_args()
    
    # Resolve worker counts
    redis_workers = args.redis_workers or args.workers or max(1, cpu_count() - 1)
    embed_workers = args.embed_workers or get_default_embed_workers()
    
    # Get vector dimensions
    if args.use_cohere:
        vector_dims = 1024
    else:
        from sentence_transformers import SentenceTransformer
        print(f"🔍 Loading HuggingFace model to get dimensions...")
        model = SentenceTransformer(args.hf_model)
        vector_dims = model.get_sentence_embedding_dimension()
        del model  # Free memory
        print(f"   Model: {args.hf_model} ({vector_dims}-dim)")
    
    config = IngestConfig(
        use_cohere=args.use_cohere,
        hf_model=args.hf_model,
        vector_dims=vector_dims,
        max_passages=args.max_passages,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        embed_workers=embed_workers,
        redis_workers=redis_workers,
    )
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"   Embedding:      {'Cohere (pre-computed)' if config.use_cohere else f'HuggingFace ({config.hf_model})'}")
    print(f"   Vector dims:    {config.vector_dims}")
    if not config.use_cohere:
        print(f"   Embed workers:  {config.embed_workers}")
    print(f"   Redis workers:  {config.redis_workers}")
    print(f"   Max passages:   {config.max_passages:,}")
    print(f"   Batch size:     {config.batch_size}")
    print(f"{'='*60}\n")
    
    total_start = time.perf_counter()
    
    # 1. Load dataset
    print(f"📂 Loading MS MARCO v2.1...")
    
    if config.use_cohere:
        ds = load_dataset(
            "Cohere/msmarco-v2.1-embed-english-v3",
            "passages",
            data_files={"train": "passages_parquet/msmarco_v2.1_doc_segmented_00.parquet"},
            split="train",
        ).select_columns(["docid", "title", "segment", "url", "emb"])
    else:
        ds = load_dataset(
            "Cohere/msmarco-v2.1-embed-english-v3",
            "passages",
            data_files={"train": "passages_parquet/msmarco_v2.1_doc_segmented_00.parquet"},
            split="train",
        ).select_columns(["docid", "title", "segment", "url"])
    
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
    embed_elapsed = 0
    
    if not config.use_cohere:
        print(f"\n🔢 Computing embeddings with {config.hf_model}...")
        texts = [row["segment"] for row in ds]
        
        embed_start = time.perf_counter()
        embeddings = embed_with_hf(
            texts,
            config.hf_model,
            config.embed_workers,
            config.encode_batch_size,
        )
        embed_elapsed = time.perf_counter() - embed_start
        print(f"   ✅ Embedded in {embed_elapsed:.1f}s ({len(texts) / embed_elapsed:.0f}/sec)")
    
    # 4. Prepare data for multiprocessing
    print(f"\n📦 Preparing batches...")
    
    data = []
    for i, row in enumerate(tqdm(ds, desc="Preparing")):
        doc = {
            "docid": row["docid"],
            "title": row["title"],
            "segment": row["segment"],
            "url": row["url"],
        }
        if config.use_cohere:
            doc["emb"] = row["emb"]
        else:
            doc["emb"] = embeddings[i].tolist()
        data.append(doc)
    
    batches = batch_list(data, config.batch_size)
    print(f"   ✅ {len(batches)} batches ready")
    
    # 5. Write to Redis
    write_start = time.perf_counter()
    
    if config.redis_workers == 1:
        print(f"\n📥 Writing to Redis (single process)...")
        
        global _redis_client, _config
        _config = config
        _redis_client = create_redis_client(config)
        
        total_docs = 0
        for batch in tqdm(batches, desc="Writing"):
            count, _ = write_batch_worker(batch)
            total_docs += count
    else:
        print(f"\n📥 Writing to Redis ({config.redis_workers} workers)...")
        
        ctx = get_context(config.mp_start_method)
        config_dict = {
            "redis_host": config.redis_host,
            "redis_port": config.redis_port,
            "redis_password": config.redis_password,
            "index_name": config.index_name,
            "index_prefix": config.index_prefix,
            "use_cohere": config.use_cohere,
            "hf_model": config.hf_model,
            "vector_dims": config.vector_dims,
            "batch_size": config.batch_size,
            "encode_batch_size": config.encode_batch_size,
            "max_passages": config.max_passages,
            "embed_workers": config.embed_workers,
            "redis_workers": config.redis_workers,
            "mp_start_method": config.mp_start_method,
        }
        
        with ctx.Pool(
            processes=config.redis_workers,
            initializer=init_redis_worker,
            initargs=(config_dict,),
        ) as pool:
            results = list(tqdm(
                pool.imap_unordered(write_batch_worker, batches),
                total=len(batches),
                desc="Writing",
            ))
        
        total_docs = sum(count for count, _ in results)
    
    write_elapsed = time.perf_counter() - write_start
    total_elapsed = time.perf_counter() - total_start
    
    # 6. Summary
    print(f"\n{'='*60}")
    print(f"✅ DONE!")
    print(f"   Total documents: {total_docs:,}")
    if embed_elapsed > 0:
        print(f"   Embed time:      {embed_elapsed:.1f}s ({total_docs / embed_elapsed:.0f}/sec)")
    print(f"   Write time:      {write_elapsed:.1f}s ({total_docs / write_elapsed:.0f}/sec)")
    print(f"   Total time:      {total_elapsed:.1f}s")
    print(f"   Overall rate:    {total_docs / total_elapsed:.0f} docs/sec")
    print(f"{'='*60}")
    
    print(f"\n📊 Index info:")
    info = index.info()
    print(f"   num_docs: {info.get('num_docs', 'N/A')}")


if __name__ == "__main__":
    main()