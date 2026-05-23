# Hot Dog Not Hot Dog

`uv run --with jupyter jupyter lab`

## Results

Zero-shot classification on Food-101 (101,000 images, 101 classes) using cosine similarity between image and text embeddings.

| Model | Overall Accuracy | Hot Dog Recall | Hot Dog Precision |
| --- | --- | --- | --- |
| CLIP ViT-B/32 | 77.9% | 73.1% | 96.4% |
| SigLIP base-patch16-224 | 89.7% | 92.9% | 95.5% |
| SigLIP2 base-patch16-224 | 90.0% | 95.5% | 92.5% |
| SigLIP2 large-patch16-256 | — | — | — |
| SigLIP2 so400m-patch14-384 | — | — | — |

SigLIP and SigLIP2 both substantially outperform CLIP. SigLIP2 requires `"a photo of a {label}"` prompt templates for best results — bare label names degrade accuracy to ~47%. Results for the large and so400m SigLIP2 variants are pending.

## References

- [The Food-101 Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
