# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project that classifies images as "hot dog" or "not hot dog" using OpenAI's CLIP (Contrastive Language-Image Pre-training) model. It leverages the Food-101 dataset (101,000 images across 101 food categories) to generate embeddings and perform zero-shot classification via cosine similarity.

The project is notebook-centric — the full ML pipeline lives in [hotdog.ipynb](hotdog.ipynb).

## Commands

```bash
uv sync                                  # Install dependencies
uv run --with jupyter jupyter lab        # Launch Jupyter Lab (or: bash run.sh)
uv run python hotdog.py                  # Extract Food-101 tarball and populate DB
uv run python schema.py                  # Initialize the SQLite database
```

## Architecture

### Pipeline (hotdog.ipynb)

The notebook is the authoritative ML workflow and runs top-to-bottom:

1. **Setup** — loads CLIP `ViT-B/32`, calls `initial_setup()` to populate `embedding.db` from the Food-101 tarball, reads env vars (`DB`, `FOOD_TAR_PATH`)
2. **Embedding generation** — encodes every image with CLIP's vision encoder; stores 512-float vectors as BLOBs in `embeddings.embedding`
3. **Text features** — encodes all 101 human-readable food labels (`labels.txt`) with CLIP's text encoder
4. **Prediction** — for each image embedding, picks the label whose text embedding has the highest cosine similarity; writes results to `predictions` table
5. **Evaluation** — overall accuracy (~78%), per-class false-negative/false-positive breakdowns, binary confusion matrix (hot dog vs. not hot dog)

### Supporting modules

- **[hotdog.py](hotdog.py)** — `initial_setup()`: opens the `.tar.gz`, extracts JPG paths, parses `id` and `label` from path structure (`food-101/images/<label>/<id>.jpg`), inserts rows into `embeddings` via `INSERT OR IGNORE`
- **[schema.py](schema.py)** — `setup_db()`: creates `embeddings` (id, path, label, embedding BLOB) and `predictions` (id, predicted_label) tables idempotently

### Database

`embedding.db` (~416 MB, not in git) is the central store. Embeddings are stored as `struct.pack('512f', ...)` BLOBs and unpacked with `struct.unpack` on read. The `predictions` table is joined back to `embeddings` for evaluation.

### Environment

`.env` file (excluded from git) must define:

- `FOOD_TAR_PATH` — path to the Food-101 `.tar.gz` download
- `DB` — optional override for the database path (defaults to `embedding.db`)

### Device detection

The notebook auto-selects `cuda` → `mps` (Apple Silicon) → `cpu` for both the CLIP model and tensor operations.

## References

- [The Food-101 Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
