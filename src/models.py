# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

from . import config


# =============================================================================
# КЛАССИФИЦИРУЮЩИЕ ГОЛОВЫ
# =============================================================================


class ClassificationHead(nn.Module):
    """
    Классифицирующая голова для одной задачи.
    Преобразует эмбеддинг в логиты целевого класса.
    """

    def __init__(
        self, input_dim: int, num_classes: int, dropout: float = config.HEAD_DROPOUT
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class MultiTaskHeads(nn.Module):
    """
    Набор из трёх классифицирующих голов для многозадачного обучения.
    """

    def __init__(
        self,
        input_dim: int,
        num_artists: int,
        num_styles: int,
        num_eras: int,
        dropout: float = config.HEAD_DROPOUT,
    ):
        super().__init__()
        self.artist_head = ClassificationHead(input_dim, num_artists, dropout)
        self.style_head = ClassificationHead(input_dim, num_styles, dropout)
        self.era_head = ClassificationHead(input_dim, num_eras, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "artist": self.artist_head(x),
            "style": self.style_head(x),
            "era": self.era_head(x),
        }

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Возвращает вероятности вместо логитов."""
        logits = self.forward(x)
        return {
            "artist": torch.softmax(logits["artist"], dim=1),
            "style": torch.softmax(logits["style"], dim=1),
            "era": torch.softmax(logits["era"], dim=1),
        }


# =============================================================================
# ГИБРИДНЫЕ АРХИТЕКТУРЫ
# =============================================================================


class SequentialHybridModel(nn.Module):
    """
    Последовательная гибридная архитектура:
    Изображение → EfficientNet → Адаптер → Swin Transformer → Классификаторы
    """

    def __init__(
        self,
        num_artists: int,
        num_styles: int,
        num_eras: int,
        cnn_arch: str = config.CNN_ARCH,
        transformer_arch: str = config.TRANSFORMER_ARCH,
        pretrained: bool = True,
        embedding_dim: Optional[int] = None,
    ):
        super().__init__()

        self.cnn = timm.create_model(
            cnn_arch, pretrained=pretrained, features_only=True, out_indices=(4,)
        )
        cnn_out_channels = self.cnn.feature_info.channels()[-1]

        self.cnn_to_transformer_adapter = nn.Sequential(
            nn.Conv2d(cnn_out_channels, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        self.transformer = timm.create_model(
            transformer_arch, pretrained=pretrained, num_classes=0
        )
        target_dim = (
            embedding_dim
            if embedding_dim is not None
            else self.transformer.num_features
        )

        self.heads = MultiTaskHeads(
            input_dim=target_dim,
            num_artists=num_artists,
            num_styles=num_styles,
            num_eras=num_eras,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. CNN + Адаптер
        cnn_features = self.cnn(x)[-1]
        adapted = self.cnn_to_transformer_adapter(cnn_features)
        x_up = F.interpolate(
            adapted, size=(224, 224), mode="bilinear", align_corners=False
        )

        # 2. Transformer
        transformer_features = self.transformer.forward_features(x_up)

        # 3. Нормализация layout [B, C, H, W]
        if (
            transformer_features.dim() == 4
            and transformer_features.shape[1] != self.transformer.num_features
        ):
            transformer_features = transformer_features.permute(0, 3, 1, 2).contiguous()

        # 4. Пулинг + Классификация
        features = transformer_features.mean(dim=(2, 3))
        return self.heads(features)

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cnn_features = self.cnn(x)[-1]
        adapted = self.cnn_to_transformer_adapter(cnn_features)
        return {
            "cnn_features": cnn_features.detach(),
            "adapted_features": adapted.detach(),
        }

    def freeze_cnn(self, freeze: bool = True):
        for param in self.cnn.parameters():
            param.requires_grad = not freeze

    def freeze_transformer(self, freeze: bool = True):
        for param in self.transformer.parameters():
            param.requires_grad = not freeze


class ParallelHybridModel(nn.Module):
    """
    Параллельная гибридная архитектура:
    Изображение → [EfficientNet || Swin Transformer] → Конкатенация → Fusion → Классификаторы
    """

    def __init__(
        self,
        num_artists: int,
        num_styles: int,
        num_eras: int,
        cnn_arch: str = config.CNN_ARCH,
        transformer_arch: str = config.TRANSFORMER_ARCH,
        pretrained: bool = True,
        num_for_increase: float = 1.0,
    ):
        super().__init__()

        self.cnn = timm.create_model(
            cnn_arch, pretrained=pretrained, features_only=False, num_classes=0
        )
        cnn_out_dim = self.cnn.num_features

        self.transformer = timm.create_model(
            transformer_arch, pretrained=pretrained, features_only=False, num_classes=0
        )
        transformer_out_dim = self.transformer.num_features

        combined_dim = cnn_out_dim + transformer_out_dim
        fusion_in_dim = int(combined_dim * num_for_increase)

        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_in_dim),
            nn.BatchNorm1d(fusion_in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(fusion_in_dim, config.EMBEDDING_DIM),
            nn.BatchNorm1d(config.EMBEDDING_DIM),
            nn.ReLU(inplace=True),
        )

        self.heads = MultiTaskHeads(
            input_dim=config.EMBEDDING_DIM,
            num_artists=num_artists,
            num_styles=num_styles,
            num_eras=num_eras,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. Извлечение признаков
        cnn_feat = self.cnn.forward_features(x)
        swin_feat = self.transformer.forward_features(x)

        # 2. Приведение к [B, C, H, W]
        if cnn_feat.dim() == 4 and cnn_feat.shape[1] < cnn_feat.shape[-1]:
            cnn_feat = cnn_feat.permute(0, 3, 1, 2).contiguous()
        if swin_feat.dim() == 4 and swin_feat.shape[1] < swin_feat.shape[-1]:
            swin_feat = swin_feat.permute(0, 3, 1, 2).contiguous()

        # 3. Глобальный пулинг
        cnn_pool = cnn_feat.mean(dim=(2, 3))
        swin_pool = swin_feat.mean(dim=(2, 3))

        # 4. Fusion
        combined = torch.cat([cnn_pool, swin_pool], dim=1)
        fused = self.feature_fusion(combined)
        return self.heads(fused)

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "cnn_features": self.cnn.forward_features(x).detach(),
            "transformer_features": self.transformer.forward_features(x).detach(),
        }

    def freeze_cnn(self, freeze: bool = True):
        for param in self.cnn.parameters():
            param.requires_grad = not freeze

    def freeze_transformer(self, freeze: bool = True):
        for param in self.transformer.parameters():
            param.requires_grad = not freeze


# =============================================================================
# ФАБРИКА МОДЕЛЕЙ
# =============================================================================


def create_model(
    model_type: str,
    num_artists: int,
    num_styles: int,
    num_eras: int,
    pretrained: bool = True,
) -> nn.Module:
    """Фабрика для создания моделей."""
    if model_type == "sequential":
        return SequentialHybridModel(
            num_artists=num_artists,
            num_styles=num_styles,
            num_eras=num_eras,
            pretrained=pretrained,
        )
    elif model_type == "parallel":
        return ParallelHybridModel(
            num_artists=num_artists,
            num_styles=num_styles,
            num_eras=num_eras,
            pretrained=pretrained,
        )
    else:
        raise ValueError(
            f"Неизвестный тип модели: {model_type}. Используйте 'sequential' или 'parallel'."
        )


# =============================================================================
# УТИЛИТЫ ДЛЯ РАБОТЫ С МОДЕЛЯМИ
# =============================================================================


def count_parameters(model: nn.Module) -> Dict[str, Any]:
    """Подсчитывает количество параметров модели."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    components = {}
    for name, module in model.named_children():
        comp_params = sum(p.numel() for p in module.parameters())
        comp_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        components[name] = {"total": comp_params, "trainable": comp_trainable}

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "components": components,
    }


def save_model_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    save_path: Path,
    extra_info: Optional[Dict] = None,
):
    """Сохраняет чекпоинт модели, оптимизатора и метаданные."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "extra_info": extra_info or {},
    }

    torch.save(checkpoint, save_path)
    print(f"Чекпоинт сохранён: {save_path}")


def load_model_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, load_path: Path
) -> Tuple[int, float, Dict]:
    """Загружает чекпоинт модели и возвращает состояние обучения."""
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Чекпоинт загружен: {load_path}")
    print(f"Эпоха: {checkpoint['epoch']}, Val Loss: {checkpoint['val_loss']:.4f}")

    return checkpoint["epoch"], checkpoint["val_loss"], checkpoint.get("extra_info", {})


def get_model_size_mb(model: nn.Module) -> float:
    """Вычисляет размер модели в мегабайтах."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def freeze_layers_by_epoch(
    model: nn.Module,
    current_epoch: int,
    freeze_cnn_until: int = 0,
    freeze_transformer_until: int = 0,
):
    """Управляет заморозкой слоёв в зависимости от текущей эпохи."""
    if hasattr(model, "freeze_cnn"):
        model.freeze_cnn(freeze=current_epoch < freeze_cnn_until)
    if hasattr(model, "freeze_transformer"):
        model.freeze_transformer(freeze=current_epoch < freeze_transformer_until)


def print_model_summary(
    model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)
):
    """Выводит сводную информацию о модели в консоль."""
    print(" СВОДКА ПО АРХИТЕКТУРЕ МОДЕЛИ")
    print(f"Тип модели: {model.__class__.__name__}")

    params = count_parameters(model)
    print(f"\n📊 Параметры:")
    print(f"  Всего: {params['total']:,}")
    print(f"  Обучаемые: {params['trainable']:,}")
    print(f"  Замороженные: {params['non_trainable']:,}")
    print(f"  Доля обучаемых: {params['trainable'] / params['total'] * 100:.1f}%")

    print(f"\n Параметры по компонентам:")
    for comp_name, comp_params in params["components"].items():
        print(f"  {comp_name}:")
        print(f"    Всего: {comp_params['total']:,}")
        print(f"    Обучаемые: {comp_params['trainable']:,}")

    print(f"\n🧪 Тестовый проход (input_size={input_size}):")
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)

    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f" Выход:")
        for task, tensor in output.items():
            print(f"    {task}: {tensor.shape}")
        print("\n Тестовый проход успешен!")
    except Exception as e:
        print(f"\n Ошибка при тестовом проходе: {e}")
