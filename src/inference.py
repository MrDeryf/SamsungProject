
import json
from pathlib import Path


import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models import ParallelHybridModel, SequentialHybridModel

# =============================================================================
# КОНСТАНТЫ ИЗ ВАШЕГО НОУТБУКА
# =============================================================================
IMG_SIZE = 224
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRAD_CAM_ALPHA = 0.4

# =============================================================================
# 1. ПРЕДОБРАБОТКА
# =============================================================================
def get_inference_transforms():
    """Возвращает пайплайн трансформаций для инференса"""
    return A.Compose([
        A.LongestMaxSize(max_size=IMG_SIZE + 32, p=1.0),
        A.PadIfNeeded(min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=cv2.BORDER_REPLICATE, p=1.0),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD, max_pixel_value=255.0),
        ToTensorV2()
    ], p=1.0)

# =============================================================================
# 2. ЗАГРУЗКА МОДЕЛИ
# =============================================================================

def load_model(weights_path: str, mappings_dir: str, model_type: str = 'parallel'):
    """Загружает веса, маппинги классов и инициализирует модель"""
    # Загрузка маппингов ID -> Name
    id2artist = json.load(open(Path(mappings_dir) / "id2artist.json", 'r', encoding='utf-8'))
    id2style  = json.load(open(Path(mappings_dir) / "id2style.json",  'r', encoding='utf-8'))
    id2era    = json.load(open(Path(mappings_dir) / "id2era.json",    'r', encoding='utf-8'))

    num_artists = len(id2artist)
    num_styles  = len(id2style)
    num_eras    = len(id2era)

    # ⚠️ ВАЖНО: Импортируйте ваши классы моделей из существующего кода
    # from your_model_module import ParallelHybridModel, SequentialHybridModel
    if model_type == 'parallel':
        model = ParallelHybridModel(num_artists, num_styles, num_eras, pretrained=False)
    else:
        model = SequentialHybridModel(num_artists, num_styles, num_eras, pretrained=False)

    # Загрузка весов (поддерживает форматы чекпоинтов из вашего Trainer)
    checkpoint = torch.load(weights_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model, id2artist, id2style, id2era

# =============================================================================
# 3. ИНФЕРЕНС
# =============================================================================
def run_inference(model, img_rgb: np.ndarray, transforms, id2artist, id2style, id2era):
    """Принимает RGB-массив, возвращает предсказания и вероятности"""
    transformed = transforms(image=img_rgb)
    input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_tensor)
        # Преобразуем логиты в вероятности
        probs = {task: torch.softmax(outputs[task], dim=1).cpu().numpy()[0] for task in outputs}

    results = {}
    for task, prob in probs.items():
        idx = int(prob.argmax())
        conf = float(prob[idx])
        if task == 'artist': results['artist'] = (id2artist[str(idx)], conf)
        elif task == 'style': results['style']  = (id2style[str(idx)],  conf)
        elif task == 'era':   results['era']    = (id2era[str(idx)],    conf)

    return results

# =============================================================================
# 4. GRAD-CAM
# =============================================================================
class GradCAMGenerator:
    def __init__(self, model):
        self.model = model
        self.target_layer = self._find_last_conv_layer()
        self.activations = None
        self.gradients = None
        self.handles = []

        # Регистрируем хуки для извлечения активаций и градиентов
        self.handles.append(self.target_layer.register_forward_hook(self._save_activations))
        self.handles.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    def _find_last_conv_layer(self):
        """Автоматически находит последний Conv2d в CNN-бэкенде"""
        for module in reversed(list(self.model.cnn.modules())):
            if isinstance(module, nn.Conv2d) and module.out_channels > 0:
                return module
        raise ValueError("Не удалось найти последний сверточный слой в CNN")

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        """Генерирует тепловую карту для входного тензора"""
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        
        # Градиент считаем относительно задачи 'artist' (можно поменять на любую)
        target = outputs['artist'][0].max()
        target.backward(retain_graph=True)

        gradients = self.gradients.cpu().numpy()[0]
        activations = self.activations.cpu().numpy()[0]

        # Взвешивание активаций средними градиентами
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Нормализация и ресайз до размера исходного изображения
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - np.min(cam)) / (np.max(cam) + 1e-7)
        return cam

    def __del__(self):
        for h in self.handles:
            h.remove()

def apply_gradcam_overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = GRAD_CAM_ALPHA):
    """Накладывает CAM на исходное изображение"""
    # Ресайзим CAM к размеру исходного изображения
    cam_resized = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    
    # Создаем heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Накладываем
    superimposed = heatmap * alpha + img_rgb.astype(np.float32) * (1 - alpha)
    return np.clip(superimposed, 0, 255).astype(np.uint8)