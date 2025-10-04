# predict.py
from cog import BasePredictor, Input, Path
import torch
import numpy as np
from scipy.ndimage import zoom
from scipy.special import logsumexp
from PIL import Image
import deepgaze_pytorch
import urllib.request
import os

class Predictor(BasePredictor):
    def setup(self):
        # escolher dispositivo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # carregar modelo pré-treinado DeepGazeIIE
        self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(self.device)
        self.model.eval()

        # fazer download do centerbias se não existir
        self.centerbias_path = "centerbias_mit1003.npy"
        if not os.path.exists(self.centerbias_path):
            print("Downloading centerbias template...")
            url = "https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy"
            urllib.request.urlretrieve(url, self.centerbias_path)
        self.centerbias_template = np.load(self.centerbias_path)

    def predict(self, image: Path = Input(description="Input image")) -> Path:
        # carregar imagem
        img = Image.open(image).convert("RGB")
        image_np = np.asarray(img).astype(np.float32) / 255.0

        # preparar centerbias
        cb = self.centerbias_template
        cb = zoom(
            cb,
            (image_np.shape[0] / cb.shape[0], image_np.shape[1] / cb.shape[1]),
            order=0,
            mode="nearest"
        )
        cb -= logsumexp(cb)  # renormalizar log density

        # converter para tensores
        image_tensor = torch.tensor([image_np.transpose(2, 0, 1)]).to(self.device)
        centerbias_tensor = torch.tensor([cb]).to(self.device)

        # inferência
        with torch.no_grad():
            log_density_prediction = self.model(image_tensor, centerbias_tensor)
            saliency = torch.exp(log_density_prediction)[0, 0].cpu().numpy()

        # normalizar e converter em imagem
        saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        saliency_img = Image.fromarray((saliency_norm * 255).astype(np.uint8))
        out_path = Path("/tmp/saliency_map.png")
        saliency_img.save(out_path)
        return out_path
