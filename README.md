<p align="center">
  <img src="https://github.com/athul-kurian/vae-latent-space-explorer/blob/main/assets/banner.gif" alt="" style="width:100%; height:auto;"/>
</p>

# VAE Latent Space Explorer

A simple tool to explore the **latent space of a Variational Autoencoder (VAE)** trained on handwritten digit images (e.g., MNIST).  
This project allows you to visualize and generate digit images by sampling or navigating the VAE’s latent space.

## 🎯 Features

- 🧠 Decode latent vectors into images using a pretrained VAE decoder
- 🖼️ Interactive GUI to explore the latent space and view generated digits in real time
- 📝 Jupyter notebook demonstrating latent space sampling and visualization

## 📁 Repository Structure

```
├── assets/                     # Static assets
├── decoder.py                  # VAE decoder implementation
├── gui.py                      # GUI for latent space exploration
├── LatentDigits.ipynb          # Notebook demo
├── decoder_weights.pt          # Pretrained decoder weights
├── .gitignore
└── README.md
```

## 🚀 Getting Started

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

### 🧪 Running the Notebook

```bash
jupyter notebook LatentDigits.ipynb
```

This notebook demonstrates:
- Loading the pretrained decoder
- Sampling points in latent space
- Visualizing generated digits

### 🖥️ Running the GUI

```bash
python gui.py
```

Use the GUI controls to modify latent variables and observe how the generated digit changes.

## 🧠 How It Works

A **Variational Autoencoder (VAE)** learns a continuous latent representation of data.  
By decoding different latent vectors, the model generates new images. Nearby points in latent space produce visually similar digits.

## 📦 Model Weights

The pretrained decoder weights are stored in:

```
decoder_weights.pt
```

These weights are loaded automatically by the decoder code.

## 📝 Example Usage

```python
from decoder import Decoder
import torch

model = Decoder()
model.load_state_dict(torch.load("decoder_weights.pt"))
model.eval()

z = torch.randn(1, 2)
img = model.decode(z)
```

## 📌 Notes

- This is an educational project intended for understanding VAEs and latent spaces
- You can extend this project to other datasets or higher-dimensional latent spaces

## 📜 License

Specify your license here.
