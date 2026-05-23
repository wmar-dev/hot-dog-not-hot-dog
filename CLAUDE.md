# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project that classifies images as "hot dog" or "not hot dog" using zero-shot classification via cosine similarity between image and text embeddings. It runs multiple vision-language models against the Food-101 dataset (101,000 images, 101 categories) and compares results.

## Commands

```bash
uv sync                                  # Install dependencies
uv run --with jupyter jupyter lab        # Launch Jupyter Lab (or: bash run.sh)
uv run python hotdog.py                  # Extract Food-101 tarball and populate DB
uv run python schema.py                  # Initialize the SQLite database
```

## Architecture

### Notebooks

Each notebook runs the full pipeline for one model family, top-to-bottom:

1. **Setup** — loads model, calls `initial_setup(model=MODEL_NAME)` to populate `embedding.db`, reads env vars
2. **Embedding generation** — encodes every image; stores float vectors as BLOBs keyed by `(id, model)`
3. **Text features** — encodes all 101 food labels from `labels.txt`
4. **Prediction** — batch cosine similarity (image matrix × text matrix); writes to `predictions` table
5. **Evaluation** — overall accuracy, per-class FN/FP breakdowns, binary confusion matrix

| Notebook | Model | Notes |
| --- | --- | --- |
| [clip.ipynb](clip.ipynb) | CLIP `ViT-B/32` | 512-dim embeddings; bare label names |
| [siglip.ipynb](siglip.ipynb) | `google/siglip-base-patch16-224` | 768-dim; bare label names work fine |
| [siglip2.ipynb](siglip2.ipynb) | `google/siglip2-*` | 768/1024/1152-dim; **requires** `"a photo of a {label}"` prompt template — bare labels degrade accuracy from ~90% to ~47% |

### Supporting modules

- **[hotdog.py](hotdog.py)** — `initial_setup(model)`: opens the `.tar.gz`, extracts JPG paths, parses `id` and `label` from path structure (`food-101/images/<label>/<id>.jpg`), inserts rows into `embeddings` via `INSERT OR IGNORE`
- **[schema.py](schema.py)** — `setup_db()`: creates `embeddings` (id, model, path, label, embedding BLOB) and `predictions` (id, model, predicted_label) tables; enables WAL journal mode

### Database

`embedding.db` (not in git) is the central store with WAL journaling. Both tables use a composite PK of `(id, model)` so multiple models coexist. Embeddings are stored as `struct.pack(f'{dim}f', ...)` BLOBs — dimension varies by model. The `predictions` table is joined back to `embeddings` for evaluation.

### Environment

`.env` file (excluded from git) must define:

- `FOOD_TAR_PATH` — path to the Food-101 `.tar.gz` download
- `DB` — optional override for the database path (defaults to `embedding.db`)

### Device detection

All notebooks auto-select `cuda` → `mps` (Apple Silicon) → `cpu`.

## References

- [The Food-101 Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224)
