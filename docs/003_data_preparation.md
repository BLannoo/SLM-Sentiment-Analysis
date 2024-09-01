# 3 Data Preparation

The dataset used for the experiments is a subset of the
[IMDb movie reviews dataset](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews),
saved in the repository to avoid compatibility issues and to provide a stable, cached version.

## 3.1 Background

The dataset is cached at [`data/imdb_subset/N=2000.csv`](data/imdb_subset/N=2000.csv) in the repository.

This approach avoids issues caused by version incompatibilities between the `datasets` Python library
and PyTorch/PyTorchVision on Google Colab.

Caching the data also allows users to directly review the recommendations.

## 3.2 Dataset and Evaluation

The dataset contains 2,000 reviews, interleaved between positive and negative sentiments.
This interleaving ensures that most subsets used for model evaluation will also be balanced.

The initial plan was to use 1,000 reviews for training and 1,000 for evaluation to achieve high accuracy,
but GPU time limitations on Colab made it infeasible to run more than 100 reviews.
Due to this, the same set was used for both iterative development and evaluation, which may lead
to some indirect data leakage by refining prompts based on observed patterns.

## 3.3 Preparing Data Locally

To recreate or modify the cached dataset, use the [cache_data.py](../src/pre/cache_data.py) script:

```bash
python cache_data.py
```

## 3.4 Note on Dataset Size

The dataset is approximately 2.6MB. While this is manageable and useful for sharing
and reproducibility in this project, it is relatively large for a Git repository, where smaller
files are preferred for efficiency.
