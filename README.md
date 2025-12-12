# MS MARCO v2.1 → Redis Vector Search PoC

Ingest MS MARCO v2.1 passages into Redis with vector embeddings, then query via RedisVL.

## Quick Start

```bash
# 1. Start Redis Stack
docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest

# 2. Install dependencies
uv sync

# 3. Run ingestion (pick one)
uv run python ingest.py --use-cohere           # Pre-computed embeddings (fast)
uv run python ingest.py --use-hf               # Local HuggingFace model

# 4. Open notebook for queries
uv run jupyter notebook queries.ipynb
```

## Embedding Options

Both ingestion scripts support two embedding modes:

| Mode | Flag | Dimensions | Speed | Notes |
|------|------|------------|-------|-------|
| **Cohere** | `--use-cohere` | 1024 | Fast | Pre-computed embeddings from dataset |
| **HuggingFace** | `--use-hf` | 384* | Slower | Local model, flexible |

\* Default HF model is `all-MiniLM-L6-v2` (384-dim). Use `--hf-model` to change.

## Ingestion Scripts

### `ingest.py` - Simple Version

Single-process ingestion. Good for quick testing.

```bash
# Pre-computed Cohere embeddings
uv run python ingest.py --use-cohere
uv run python ingest.py --use-cohere --max-passages 50000

# Local HuggingFace embeddings
uv run python ingest.py --use-hf
uv run python ingest.py --use-hf --hf-model sentence-transformers/all-mpnet-base-v2
```

### `ingest_multi.py` - Multiprocessing Version

Parallel ingestion with separate controls for embedding and Redis workers.

```bash
# Cohere + parallel Redis writes
uv run python ingest_multi.py --use-cohere --workers 8

# HuggingFace + separate worker controls
uv run python ingest_multi.py --use-hf --embed-workers 1 --redis-workers 8
```

**CLI Options:**

| Flag | Description | Default |
|------|-------------|---------|
| `--use-cohere` | Use pre-computed Cohere embeddings | - |
| `--use-hf` | Use local HuggingFace model | - |
| `--hf-model` | HuggingFace model name | `all-MiniLM-L6-v2` |
| `--workers` | Workers for Redis writes (shorthand) | CPU count - 1 |
| `--redis-workers` | Workers for Redis writes | CPU count - 1 |
| `--embed-workers` | Workers for HF embedding | Auto-detect |
| `--max-passages` | Maximum passages to ingest | 100,000 |
| `--batch-size` | Batch size for Redis writes | 500 |
| `--encode-batch-size` | Batch size for HF encoding | 128 |

---

## Queries Notebook

The `queries.ipynb` notebook demonstrates vector search, full-text search, and hybrid search.

### ⚠️ Important: Match Your Embedding Mode

The notebook has **two configuration cells** at the top. **Run only one** based on how you ingested the data:

| If you ingested with... | Run this cell in notebook |
|-------------------------|---------------------------|
| `--use-cohere` | **Option A: Cohere** |
| `--use-hf` | **Option B: HuggingFace** |

**Why?** The vector dimensions must match:
- Cohere embeddings are 1024-dim
- HuggingFace (default) embeddings are 384-dim

Running the wrong cell will cause dimension mismatch errors.

### Notebook Sections

1. **Vector Search** - Semantic similarity using embeddings
2. **Full-Text Search** - Traditional keyword matching
3. **Hybrid Search** - Vector + text/metadata filters
4. **Batch Evaluation** - QPS and latency benchmarks

---

## Hardware Auto-Detection (for `--use-hf`)

| Hardware | Embed Workers | Notes |
|----------|---------------|-------|
| Multi-GPU (CUDA) | GPU count | 1 process per GPU |
| Single GPU | 1 | Multiprocessing adds overhead |
| Apple MPS | 1 | MPS can't be shared across processes |
| CPU only | CPU count - 1 | Bypasses Python GIL |

**Platform Examples:**

```bash
# Mac with Apple Silicon (MPS)
uv run python ingest_multi.py --use-hf --embed-workers 1 --redis-workers 8

# Multi-GPU server (4 GPUs)
uv run python ingest_multi.py --use-hf --embed-workers 4 --redis-workers 16

# CPU-only machine
uv run python ingest_multi.py --use-hf --embed-workers 4 --redis-workers 8

# Just want fast ingestion? Use Cohere
uv run python ingest_multi.py --use-cohere --workers 8
```

---

## Dataset

Uses [Cohere/msmarco-v2.1-embed-english-v3](https://huggingface.co/datasets/Cohere/msmarco-v2.1-embed-english-v3) from HuggingFace.

- **Passages**: Single shard `msmarco_v2.1_doc_segmented_00.parquet` (~700k passages)
- **Queries**: Test split with pre-computed embeddings (used in notebook)

## Schema

Each document in Redis contains:

| Field | Type | Description |
|-------|------|-------------|
| `docid` | tag | Unique document ID |
| `title` | text | Document title (full-text searchable) |
| `segment` | text | Passage content (full-text searchable) |
| `url` | tag | Source URL |
| `text_embedding` | vector | HNSW cosine (1024-dim Cohere or 384-dim HF) |

---

## Reset Environment

```bash
# Reset uv
rm -rf .venv uv.lock
uv sync

# Reset Redis
docker rm -f redis-stack-server
docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest
```

## Redis Insight

For a UI to explore your data, use `redis/redis-stack` instead (includes RedisInsight):

```bash
docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

Then open http://localhost:8001