# Hot Dog Not Hot Dog

`uv run --with jupyter jupyter lab`

## Results

Zero-shot classification on Food-101 (101,000 images, 101 classes) using cosine similarity between image and text embeddings.

| Model | Overall Accuracy | Hot Dog Recall | Hot Dog Precision |
| --- | --- | --- | --- |
| CLIP ViT-B/32 | 77.9% | 73.1% | 96.4% |
| SigLIP base-patch16-224 | 89.7% | 92.9% | 95.5% |

SigLIP improves overall accuracy by ~12 points and hot dog recall by ~20 points, with a small drop in precision.

## References

- [The Food-101 Data Set](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)
