# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project that classifies images as "hot dog" or "not hot dog" using OpenAI's CLIP (Contrastive Language-Image Pre-training) model. It leverages the Food-101 dataset (101,000 images across 101 food categories) to generate embeddings and perform classification via semantic similarity matching.

The project is notebook-centric, with the main ML workflow in [hotdog.ipynb](hotdog.ipynb).

## Common Development Commands

**Setup and Dependencies:**
```bash
uv sync              # Install dependencies from pyproject.toml
```

**Running the ML Workflow:**
```bash
uv run --with jupyter jupyter lab    # Launch Jupyter Lab to work with hotdog.ipynb
```

**Running Python Scripts:**
```bash
uv run python hotdog.py    # Extract Food-101 dataset and populate database
uv run python main.py      # Run the main entry point
uv run python schema.py    # Initialize or inspect database schema
```

## Architecture Overview

### Core Components

1. **[hotdog.ipynb](hotdog.ipynb)** - Main workflow notebook
   - Loads CLIP model (vision + text encoders)
   - Generates embeddings for all Food-101 images
   - Trains on embeddings using cosine similarity
   - Performs predictions and evaluation
   - Contains 25 code cells with complete ML pipeline

2. **[hotdog.py](hotdog.py)** - Data extraction utility
   - Reads the Food-101 tarball (specified by `FOOD_TAR_PATH` env var)
   - Extracts JPG files and metadata
   - Populates SQLite database with image paths and labels

3. **[schema.py](schema.py)** - Database schema
   - Defines `embeddings` table: id, path, label, embedding (BLOB)
   - Defines `predictions` table: id, predicted_label
   - Initializes SQLite database

4. **[union_find.py](union_find.py)** - Utility data structure
   - Union-Find (Disjoint Set Union) implementation
   - Used for grouping similar images or clustering operations

### Data Organization

- **[classes.txt](classes.txt)** - 101 food class names (snake_case format)
- **[labels.txt](labels.txt)** - 101 food class names (human-readable format)
- **[train.txt](train.txt)** - Training split (75,750 samples), format: `category_name/image_id`
- **[test.txt](test.txt)** - Test split (25,250 samples)
- **[embedding.db](embedding.db)** - SQLite database (416 MB) storing embeddings and predictions

### Key Design Decisions

**Embedding-Based Classification:**
- Uses pre-trained CLIP model to generate fixed-size image embeddings
- Performs classification via cosine similarity (zero-shot learning approach)
- Avoids training custom neural networks from scratch

**Persistent Storage:**
- SQLite stores computed embeddings as BLOBs to avoid recomputation
- Allows quick iteration on classification logic without regenerating embeddings

**Environment Configuration:**
- `FOOD_TAR_PATH` environment variable specifies the Food-101 dataset tarball location
- Defined in `.env` file (excluded from git)

**Device Agnostic:**
- Automatically detects CUDA (NVIDIA GPU), Apple Metal (Apple Silicon), or falls back to CPU
- CLIP models loaded with appropriate device specifications

## Dependencies

Core ML libraries specified in [pyproject.toml](pyproject.toml):
- **torch** (>=2.9.1) - Deep learning framework
- **clip** (>=0.2.0) - OpenAI's vision-language model
- **pandas** (>=2.3.3) - Data manipulation
- **matplotlib** (>=3.10.7) - Visualization
- **jupyter** (>=1.1.1) - Interactive notebooks
- **dotenv** (>=0.9.9) - Environment variable management

Minimum Python version: **3.14+**

## Testing and Evaluation

The notebook contains evaluation cells that:
- Compute accuracy on the test split
- Calculate cosine similarity scores between images
- Validate predictions against ground truth labels

Run evaluation cells within [hotdog.ipynb](hotdog.ipynb) after training to assess model performance.

## References

- [The Food-101 Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- OpenAI CLIP: https://github.com/openai/CLIP
