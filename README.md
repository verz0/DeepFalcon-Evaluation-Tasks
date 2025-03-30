# DeepFalcon-Evaluation-Tasks

## Graph Representation Learning for Fast Detector Simulation

This repository contains the notebooks and models for the tasks I have completed for the project "Graph Representation Learning for Fast Detector Simulation" for GSoC 2025

## Tasks I Completed

I've structured the repository based on the tasks I completed:

1.  **Common Task 1. Auto-encoder of the quark/gluon events (`common1/`)**
    * Implemented and trained a VAE using CNNs to learn compressed representations of the 3-channel jet images.
    * My focus was on reconstructing the input images from the learned latent space.
    * Generated visualizations showing side-by-side comparisons of original and reconstructed jet events.
    * Analyzed the VAE's latent space using PCA for dimensionality reduction and visualization, coloring points by their quark/gluon labels.

2.  **Common Task 2. Jets as graphs (`common2/`)**
    * I developed a pipeline to convert jet images into graph representations:
        * I first converted images to point clouds by selecting pixels above an intensity threshold.
        * For node features, I used the original 3 channel intensities plus engineered polar coordinates (radius, angle).
        * I constructed graphs using K-Nearest Neighbors (KNN) based on spatial coordinates.
        * Used the normalized distance between nodes as an edge feature.
    * Implemented and trained a Graph Attention Network (GATv2) for classifying jets as quarks or gluons.
    * Evaluated the GNN's performance using accuracy, confusion matrix, and classification reports.
    * Visualized the learned graph embeddings using UMAP to see if the classes were separable.

3.  **Specific Task 1 (if you are interested in “Graph Representation Learning for Fast Detector Simulation” project) (`specific/`)**
    * Implemented and trained a Graph Autoencoder (GAE) using Graph Convolutional Network (GCN) layers.
    * Trained the GAE to reconstruct node features (which included channel intensities and normalized coordinates).
    * Developed a method to map the reconstructed node features back to a 2D image format for visualization.
    * Compared the visual reconstruction quality and Mean Squared Error (MSE) of my GAE against the VAE from Common Task 1. The GAE achieved a lower average reconstruction MSE on the validation samples I checked.

## What I Learned

* How to apply VAEs for unsupervised representation learning and image reconstruction in a physics data context.
* Various techniques for converting structured image data into graph representations (point clouds, KNN graphs) and how to engineer relevant node/edge features (like polar coordinates or distance).
* How to implement and train different GNN architectures (GATv2 for classification, GCN for autoencoding) using PyTorch Geometric.
* The conceptual differences and potential trade-offs between using standard CNN-based autoencoders versus graph-based autoencoders for this type of jet data.

## Challenges I Faced & Considerations

* **Computational Resources:** Training these deep learning models was computationally demanding. Using GPUs was essential for feasible training times, and even then, I worked with subsets (15,000 samples) of the full dataset to manage runtimes.
* **Hyperparameter Tuning:** Finding the best settings (like learning rates, network sizes, `k` for KNN, VAE's beta value, dropout rates) wasn't straightforward and usually requires more extensive experimentation than I performed in these initial explorations.
* **Graph Construction Details:** Turning images into graphs involves making several choices (intensity threshold, how to define nodes/edges/features) that significantly influence the results. I also had to filter out graphs that didn't have enough nodes after thresholding.
* **Visualization:** Visualizing the high-dimensional outputs (latent spaces, embeddings) needed dimensionality reduction (PCA, UMAP). I also had to adjust plotting parameters like axis limits to get meaningful views, especially for the latent space plots.

PS: The respective directories and the notebooks for each task has detailed explanations included for your reference and evaluation.
    The models were developed in a Kaggle environment (16GB RAM + T4 GPU)
