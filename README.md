# MS MARCO → Redis Vector Search PoC

Ingest MS MARCO passages into Redis with vector embeddings, then query via RedisVL.

## Quick Start

```bash
# 1. Start Redis Stack
docker compose up -d

# 2. Install dependencies
uv sync

# 3. Run ingestion (100k passages by default)
uv run python ingest.py

# 4. Open notebook for queries
uv run jupyter notebook queries.ipynb
```

## Configuration

Edit `ingest.py` to adjust:

- `MAX_PASSAGES` - Number of passages to ingest (default: 100,000)
- `BATCH_SIZE` - Documents per batch (default: 500)
- `NUM_WORKERS` - Parallel embedding workers (default: CPU count - 1)
- `MODEL_NAME` - Embedding model (default: all-MiniLM-L6-v2)

## Schema

Each document contains:

| Field | Type | Description |
|-------|------|-------------|
| `passage_id` | tag | Unique ID (query_id + index) |
| `text` | text | Passage content (full-text searchable) |
| `url` | tag | Source URL |
| `embedding` | vector | 384-dim HNSW cosine |

## Redis Insight

View your data at http://localhost:8001
