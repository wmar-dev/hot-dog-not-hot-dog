# Hot Dog Not Hot Dog

Inspired by the [Silicon Valley](https://www.hbo.com/silicon-valley) bit: can a vision-language model tell a hot dog from everything else? This project runs zero-shot image classification across the full [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset — 101,000 images spanning 101 food categories — using cosine similarity between image and text embeddings. No fine-tuning, no labeled training data; just a model and a list of class names.

## Setup

```bash
uv sync                                  # install dependencies
cp .env.example .env                     # set FOOD_TAR_PATH to your Food-101 download
uv run python schema.py                  # initialize embedding.db
uv run python hotdog.py                  # populate DB with image paths
uv run --with jupyter jupyter lab        # launch notebooks
```

## How it works

Each notebook encodes all 101,000 images into embedding vectors, encodes the 101 food-class label strings, then finds the nearest label to each image via batch matrix multiplication. The prediction with the highest cosine similarity wins. No threshold tuning — the model either knows what a hot dog looks like or it doesn't.

The binary "hot dog / not hot dog" framing collapses the 101-class problem into a single judgment: was the top-1 prediction `hot_dog`?

## Results

| Model | Overall Accuracy | Hot Dog Recall | Hot Dog Precision | Hot Dog F1 |
| --- | --- | --- | --- | --- |
| CLIP ViT-B/32 | 77.9% | 73.1% | 96.4% | 83.2% |
| SigLIP base-patch16-224 | 89.7% | 92.9% | 95.5% | 94.2% |
| SigLIP2 base-patch16-224 | 90.0% | 95.5% | 92.5% | 94.0% |
| SigLIP2 large-patch16-256 | — | — | — | — |
| SigLIP2 so400m-patch14-384 | — | — | — | — |

SigLIP and SigLIP2 substantially outperform CLIP on both overall accuracy and hot dog recall. The main failure mode for CLIP is false negatives — it misses hot dogs, often confusing them with other sandwiches. SigLIP2 nearly eliminates false positives (only 2 non-hot-dog images misclassified as hot dog in the base variant), but its false negatives lean heavily toward `club_sandwich` and `pulled_pork_sandwich`.

One gotcha: SigLIP2 requires a prompt template (`"a photo of a {label}"`) for best results. Using bare label names drops overall accuracy from ~90% to ~47%. SigLIP and CLIP work fine with bare labels.

Results for the SigLIP2 large and so400m variants are pending.

## References

- [The Food-101 Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-224)
- [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224)
