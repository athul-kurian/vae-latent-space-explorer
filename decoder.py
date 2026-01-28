import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6, 32 * 7 * 7)
        self.deconv_stack = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 32, 7, 7)
        return self.deconv_stack(x)

decoder = Decoder()
decoder.load_state_dict(torch.load("decoder_weights.pt", map_location="cpu"))
decoder.eval()

PRESETS = {
    0: [-3.0000, -2.1033, -1.2717,  0.0978,  1.3533, -1.0924],
    1: [ 1.7609,  0.0000,  0.0000, -0.8152, -1.0761,  0.3587],
    2: [-0.2446, -1.3370,  2.0217, -0.7989, -0.0815,  0.1141],
    3: [-0.7663, -0.6033,  1.9239,  0.2446, -1.2717,  0.3424],
    4: [-0.0815, -2.2826, -1.3043,  1.0109,  0.7663,  0.6196],
    5: [-0.3261,  0.0000,  0.0000,  0.3750, -2.3478, -0.6522],
    6: [ 0.0000,  0.0000,  0.0000,  0.0000, -1.1739, -2.1033],
    7: [-0.1141,  1.0598,  0.5380,  0.3750,  0.9130,  2.3804],
    8: [-0.8315, -1.6141,  0.1467,  0.3750, -1.0761, -0.0326],
    9: [ 0.8152, -0.4239, -0.4728,  0.6848,  0.3261,  1.0924],
}

def latent_to_image(latent):
    z = torch.tensor([latent], dtype=torch.float32)
    with torch.no_grad():
        img = decoder(z).squeeze().numpy()

    img = (img + 1.0) / 2.0
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img, mode="L")
