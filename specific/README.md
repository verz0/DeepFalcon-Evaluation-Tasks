# Specific Task 1 (if you are interested in “Graph Representation Learning for Fast Detector Simulation” project):

This notebook focuses on implementing a Graph Autoencoder (GAE) for jet reconstruction and comparing its performance to a standard Variational Autoencoder (VAE) used in common task 1.

## What I Did

1.  **Data Loading & Preprocessing:** Loaded and preprocessed the jet data (resizing, normalization) similarly to the other notebooks. Split into train/validation sets.
2.  **VAE Loading:** Defined the VAE architecture (matching `common1/`) and included logic to load pre-trained VAE weights for later comparison.
3.  **Image-to-Graph Conversion (GAE specific):**
    * Converted images to graphs where nodes represent pixels above an intensity threshold.
    * **Node Features:** Used the 3 channel intensities PLUS normalized spatial coordinates (y, x) as the 5 node features. Including coordinates is crucial as the GAE aims to reconstruct them.
    * **Graph Structure:** Used K-Nearest Neighbors (KNN) based on spatial coordinates to define graph edges (edge_index).No edge features were used for this GAE model.
    * Created a PyTorch Geometric dataset storing these graphs.
4.  **GAE Model:** Implemented a Graph Autoencoder using GCNConv layers:
    * An **Encoder** using GCN layers to map input node features and graph structure to latent node embeddings.
    * A **Decoder** (simple MLP) to reconstruct the original 5D node features from the latent embeddings.
5.  **GAE Training:**
    * Trained the GAE using **Mean Squared Error (MSE)** loss between the original node features and the reconstructed node features.
    * Used an Adam optimizer and `ReduceLROnPlateau` scheduler based on validation MSE loss.
    * Saved the best GAE model based on validation loss.
6.  **Graph-to-Image Reconstruction:** Implemented a function to convert the GAE's output (reconstructed node features) back into a 2D image format. This involved:
    * Retrieving the original pixel coordinates stored within the `Data` object.
    * Extracting only the reconstructed energy channel features (first 3 dimensions of the output).
    * Placing these energies back onto a blank image canvas at the correct coordinates.
7.  **Comparison & Visualization:**
    * Loaded the best trained GAE and the pre-trained VAE.
    * For sample validation events, generated reconstructions using both the VAE and the GAE
    * Calculated the MSE between the original normalized image and each model's normalized reconstruction.
    * Generated side-by-side visualizations comparing the denormalized original image, VAE reconstruction, and GAE reconstruction.
    * Generated per-channel visualizations for a deeper comparison.
    * Calculated and compared the average validation MSE for both models.
  
The GAE clearly outperforms the VAE and its results can be seen:

![image](https://github.com/user-attachments/assets/4ea12b3e-4a03-4bfd-8546-7150f47c5a81)


![image](https://github.com/user-attachments/assets/ba2e91c5-3a0b-4d43-bc04-b890e771f810)


![image](https://github.com/user-attachments/assets/4aad4d04-842d-4166-acca-13243024e22f)


![image](https://github.com/user-attachments/assets/c982d26f-7756-43a9-8c84-89d0ed4aef3d)




