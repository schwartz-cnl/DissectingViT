# DissectingViT
This is the official code repository for the paper "Dissecting Query-Key Interaction in Vision Transformers" [[link](https://arxiv.org/abs/2405.14880)] by Xu Pan, Aaron Philip, Ziqian Xie, Odelia Schwartz.

Set up the path in "config.py" before running.

1. "cosine_similarity.ipynb" is for (1) saving SVD of the query-key interaction matrix. (2) plot the averaged cosine similarity between U and V. (3) plotting the singular spectrum.

2. "look_for_optimal_imagesnet_photo.ipynb" is for (1) saving files containing the projection values of imagenet image embedding onto singular vectors and attention values (product of max projection on U and max projection on V), (2) drawing figures with top attention images for each layer, head and mode, and corresponding U map and V map. 
