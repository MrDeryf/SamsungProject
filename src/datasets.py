import cv2
import pandas as pd
import json
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from . import config


# =============================================================================
# ТРАНСФОРМАЦИИ ИЗОБРАЖЕНИЙ (Albumentations)
# =============================================================================


def get_train_transforms() -> A.Compose:
    """
    Трансформации для обучающей выборки.
    Включает аугментации: RandomResizedCrop, Flip, ShiftScaleRotate.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=config.IMG_SIZE + 32, p=1.0),
            A.RandomResizedCrop(
                size=(config.IMG_SIZE, config.IMG_SIZE),
                scale=(0.8, 1.0),
                ratio=(0.95, 1.05),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.03,
                rotate_limit=10,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.5,
            ),
            A.Normalize(
                mean=config.IMAGE_NET_MEAN,
                std=config.IMAGE_NET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        p=1.0,
    )


def get_val_transforms() -> A.Compose:
    """
    Трансформации для валидационной и тестовой выборок.
    Без агрессивных аугментаций, только центральная обрезка.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=config.IMG_SIZE + 32, p=1.0),
            A.PadIfNeeded(
                min_height=config.IMG_SIZE,
                min_width=config.IMG_SIZE,
                border_mode=cv2.BORDER_REPLICATE,
                p=1.0,
            ),
            A.CenterCrop(height=config.IMG_SIZE, width=config.IMG_SIZE, p=1.0),
            A.Normalize(
                mean=config.IMAGE_NET_MEAN,
                std=config.IMAGE_NET_STD,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
        p=1.0,
    )


def get_test_transforms() -> A.Compose:
    """
    Трансформации для тестовой выборки (аналогичны валидации).
    """
    return get_val_transforms()


def get_inference_transforms() -> A.Compose:
    """
    Трансформации для инференса новых изображений (аналогичны валидации).
    """
    return get_val_transforms()


# =============================================================================
# КЛАСС ДАТАСЕТА
# =============================================================================


class ArtDataset(Dataset):
    """
    Кастомный Dataset для загрузки изображений с тремя метками:
    - artist (художник)
    - style (стиль — случайный выбор из списка художника)
    - era (эпоха — фиксированная для художника)
    """

    def __init__(
        self,
        csv_path: Path,
        artist2id: Dict[str, int],
        era2id: Dict[str, int],
        style2id: Dict[str, int],
        transforms=None,
        return_path: bool = False,
        random_seed: int = config.RANDOM_SEED,
    ):
        self.data = pd.read_csv(csv_path)
        self.artist2id = artist2id
        self.era2id = era2id
        self.style2id = style2id
        self.transforms = transforms
        self.return_path = return_path
        self.random = random.Random(random_seed)

        # Нормализация имен художников в маппинге
        normalized_mapping = {k.strip().lower(): v for k, v in self.artist2id.items()}
        self.artist2id = normalized_mapping

        # Подготовка данных: фильтрация и нормализация
        self.data["artist_clean"] = (
            self.data["artist"].astype(str).str.strip().str.lower()
        )

        # Фильтрация строк с неизвестными художниками
        original_len = len(self.data)
        self.data = self.data[
            self.data["artist_clean"].isin(self.artist2id.keys())
        ].copy()
        filtered_len = len(self.data)

        if filtered_len < original_len:
            print(
                f" Отфильтровано {original_len - filtered_len} изображений "
                f"с неизвестными художниками в {csv_path.name}"
            )

        if "artist_clean" in self.data.columns:
            self.data = self.data.drop(columns=["artist_clean"])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Dict[str, int], Optional[str]]:
        row = self.data.iloc[idx]
        img_path = Path(row["image_path"])
        artist_name = row["artist"].strip().lower()
        era = row["era"]
        style = self.random.choice(row["styles"])

        # Загрузка изображения
        image = cv2.imread(str(img_path))
        if image is None:
            # Фоллбэк на черную картинку или пропуск, здесь лучше кинуть ошибку
            # но для обучения иногда заменяют. Оставим ValueError.
            raise ValueError(f"Не удалось загрузить изображение: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Применение трансформаций
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        # Получение ID художника
        artist_id = self.artist2id[artist_name]
        era_id = self.era2id[era]
        style_id = self.style2id[style]

        labels = {"artist": artist_id, "style": style_id, "era": era_id}

        if self.return_path:
            return image, labels, str(img_path)
        else:
            return image, labels


# =============================================================================
# УТИЛИТЫ ДЛЯ ЗАГРУЗКИ ДАННЫХ И СОЗДАНИЯ DATALOADERS
# =============================================================================


def load_all_mappings(
    processed_dir: Path = config.PROCESSED_DATA_DIR,
    metadata_csv: Path = config.ARTIST_METADATA_CSV,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[Dict, Dict, Dict, Dict]:
    """
    Загружает все необходимые маппинги для Dataset из JSON файлов.
    """
    if not processed_dir.exists():
        raise FileNotFoundError(f"Директория маппингов не найдена: {processed_dir}")

    # Загрузка artist2id
    artist2id_path = processed_dir / "artist2id.json"
    if not artist2id_path.exists():
        raise FileNotFoundError(f"Файл {artist2id_path} не найден.")

    with open(artist2id_path, "r", encoding="utf-8") as f:
        artist2id = json.load(f)

    # Загрузка style2id и era2id (если есть)
    style2id_path = processed_dir / "style2id.json"
    era2id_path = processed_dir / "era2id.json"

    style2id = {}
    if style2id_path.exists():
        with open(style2id_path, "r", encoding="utf-8") as f:
            style2id = json.load(f)

    era2id = {}
    if era2id_path.exists():
        with open(era2id_path, "r", encoding="utf-8") as f:
            era2id = json.load(f)

    # Импорт функций из data_processing для создания маппинга artist -> labels,
    # если его нет готового или нужно пересоздать.
    # В идеале этот маппинг тоже должен сохраняться, но в ноутбуке он создавался динамически.
    # Для чистоты структуры вызовем функцию создания маппинга.

    # Чтобы избежать циклического импорта, импортируем функцию здесь или перенесем логику.
    # В данном случае, так как data_processing.py уже написан, импортируем.
    from .data_processing import (
        load_artist_metadata,
        create_era_and_style_mappings,
    )

    # Если маппинги стилей/эпох не найдены, создаём их (фоллбэк)
    if not style2id or not era2id:
        print("⚠️ style2id или era2id не найдены, создаём их заново...")
        metadata_df = load_artist_metadata(metadata_csv)
        (style2id, era2id), _ = create_era_and_style_mappings(
            metadata_df, artist2id, processed_dir
        )

    return artist2id, style2id, era2id


def collate_fn(batch):
    """
    Collate function для обработки батчей с многозадачными метками.
    Собирает изображения в один тензор, а метки в словарь тензоров.
    """
    if not batch:
        return torch.tensor([]), {}

    # Проверяем, есть ли пути в элементах батча
    if len(batch[0]) == 3:
        # С путями: (image, labels, path)
        images, labels_list, paths = zip(*batch)
        has_paths = True
    else:
        # Без путей: (image, labels)
        images, labels_list = zip(*batch)
        paths = None
        has_paths = False

    # Stack изображений в один тензор [B, C, H, W]
    images = torch.stack(images)

    # Конвертация списка словарей меток в словарь тензоров
    # labels_list: [{'artist': 0, 'style': 2, 'era': 1}, {...}, ...]
    labels = {
        "artist": torch.LongTensor([l["artist"] for l in labels_list]),
        "style": torch.LongTensor([l["style"] for l in labels_list]),
        "era": torch.LongTensor([l["era"] for l in labels_list]),
    }

    if has_paths:
        return images, labels, list(paths)
    else:
        return images, labels


def create_dataloaders(
    data_dir: Path,
    artist2id: Dict[str, int],
    era2id: Dict[str, int],
    style2id: Dict[str, int],
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
    pin_memory: bool = config.PIN_MEMORY,
    random_seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт DataLoaders для train/val/test с тремя метками.
    """
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    test_transforms = get_test_transforms()

    # Создание датасетов
    train_dataset = ArtDataset(
        csv_path=data_dir / "train.csv",
        artist2id=artist2id,
        era2id=era2id,
        style2id=style2id,
        transforms=train_transforms,
        random_seed=random_seed,
    )

    val_dataset = ArtDataset(
        csv_path=data_dir / "val.csv",
        artist2id=artist2id,
        era2id=era2id,
        style2id=style2id,
        transforms=val_transforms,
        random_seed=random_seed,
    )

    test_dataset = ArtDataset(
        csv_path=data_dir / "test.csv",
        artist2id=artist2id,
        era2id=era2id,
        style2id=style2id,
        transforms=test_transforms,
        return_path=True,  # Для теста нужны пути, чтобы строить отчёты
        random_seed=random_seed,
    )

    # Создание DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader


def calculate_class_weights(
    artist2id: Dict[str, int],
    style2id: Dict[str, int],
    era2id: Dict[str, int],
    train_csv: Path,
    artist_labels_mapping: Dict[int, Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    Вычисляет веса классов для всех трёх задач по Формуле: ω_c = N_total / (C * N_c).
    """
    df = pd.read_csv(train_csv)
    df["artist_clean"] = df["artist"].astype(str).str.strip().str.lower()

    weights = {}

    for task, id_mapping in [
        ("artist", artist2id),
        ("style", style2id),
        ("era", era2id),
    ]:
        N_total = len(df)
        C = len(id_mapping)

        class_counts = {}

        for _, row in df.iterrows():
            artist_name = row["artist_clean"]
            artist_id = artist2id.get(artist_name)

            if artist_id is None or artist_id not in artist_labels_mapping:
                continue

            if task == "artist":
                class_id = artist_id
            elif task == "style":
                # Для стиля: учитываем все стили художника пропорционально
                style_ids = artist_labels_mapping[artist_id]["style_ids"]
                weight_per_style = 1.0 / len(style_ids) if style_ids else 1.0
                for sid in style_ids:
                    class_counts[sid] = class_counts.get(sid, 0) + weight_per_style
                continue
            else:  # era
                class_id = artist_labels_mapping[artist_id]["era_id"]

            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        # Вычисление весов
        task_weights = [0.0] * len(id_mapping)
        for class_name, class_id in id_mapping.items():
            N_c = max(class_counts.get(class_id, 1), 1)  # Защита от 0
            task_weights[class_id] = N_total / (C * N_c)

        weights[task] = torch.FloatTensor(task_weights)

    return weights
