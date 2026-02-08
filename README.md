<p align="center">
  <img src="https://github.com/athul-kurian/vae-latent-space-explorer/blob/main/assets/banner.gif" alt="" style="width:100%; height:auto;"/>
</p>

## 🧠 Overview

A tool to explore the **latent space of a Variational Autoencoder (VAE)model** trained on handwritten digit images (MNIST). This project allows you to visualize and generate digit images by sampling or navigating the VAE’s latent space.

## 🚀 How to Use

### 🛠️ Requirements

- Python 3.7+
- PyTorch
- tkinter
- matplotlib
- numpy

Install dependencies:

```bash
pip install torch matplotlib numpy
```

> Note: `tkinter` is usually included with Python installations.

### 🖥️ Running the GUI

```bash
python gui.py
```

Use the GUI controls to modify latent variables and observe how the generated digit changes.

## 🧠 VAE Architecture

### Encoder
```
- Input image: 1x28x28
- Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1) + ReLU
  - Weight Matrix: 16 x 4 x 4
  - Output: 16 x 14 x 14
- Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1) + ReLU
  - Weight Matrix: 32 x 4 x 4
  - Ouput: 32 x 7 x 7
- Flatten layer: flattens each input from 32 x 7 x 7 to 1 x 1568
- Linear(in_features=1568, out_features=6)
  - Outputs the mean vector of the latent Gaussian distribution; The mean vector has a length of 6 corresponding to the 6 latent dimensions
  - Weight Matrix: (6, 1568)
- Linear(in_features=1568, out_features=6)
  - Ouputs the log-variance vector of the latent Gaussian distribution; We assume a dioganal co-variance matrix, so each latent dimension has an independent co-variance
```

### Reparameterization

- Instead of sampling directly from the latent distribution, we sample from a standard normal distribution and scale and shift it using the predicted mean and variance.

```
z = μ + exp(0.5*logσ²) * ε,   ε ~ N(0, 1)
```

### Decoder
```
- Linear(in_features=latent_dim, out_features=32*7*7)
- Reshape: Converts linear output to feature maps (from 1 x 1568 to 32 x 7 x 7)
- ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1) + ReLU
- ConvTranspose2d(in_channels=16, out_channels=input_dim, kernel_size=4, stride=2, padding=1) + Tanh

```

### 🧪 Training

The code used to build and train the model can be found here in [LatentDigits.ipynb](https://github.com/athul-kurian/vae-latent-space-explorer/blob/main/LatentDigits.ipynb)

Key training parameters (from the notebook):
- Dataset: torchvision MNIST (80,000 1x28x28 images)
- Preprocessing: `ToTensor()` then `Normalize((0.5,), (0.5,))` to map pixels from [0, 1] to [-1, 1] for the decoder’s `tanh` output.
- Latent dimension: 6.
- Batch size: 128, with shuffling.
- Epochs: 200.
- Optimizer: Adam with learning rate 5e-3.
- Loss: sum of pixel-wise MSE reconstruction loss + KL divergence term.
- Device: T4 GPU.

## 📦 Decoder Weights

The trained decoder weights are stored in [decoder_weights.pt](https://github.com/athul-kurian/vae-latent-space-explorer/blob/main/decoder_weights.pt)
