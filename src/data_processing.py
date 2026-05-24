import pandas as pd
import numpy as np
import cv2
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Импортируем константы
import config


def get_image_paths(data_dir: Path) -> Dict[str, List[Path]]:
    """
    Собирает пути ко всем изображениям, сгруппированные по художникам.

    Args:
        data_dir: Путь к директории с сырыми изображениями.

    Returns:
        Словарь, где ключи — имена художников, а значения — списки путей к файлам.
    """
    artist_images = {}
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Директория данных не найдена: {data_dir}")

    for artist_folder in data_dir.iterdir():
        if artist_folder.is_dir():
            artist_name = artist_folder.name
            images = [img for img in artist_folder.iterdir() if img.is_file()]
            artist_images[artist_name] = images
    return artist_images


def check_broken_images(image_paths: List[Path]) -> List[Path]:
    """
    Проверяет список файлов изображений на целостность.

    Args:
        image_paths: Список путей к файлам.

    Returns:
        Список путей к битым файлам.
    """
    broken_images = []
    for img_path in image_paths:
        try:
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                broken_images.append(img_path)
        except Exception:
            broken_images.append(img_path)
    return broken_images


def analyze_dataset(data_dir: Path) -> pd.DataFrame:
    """
    Проводит разведочный анализ датасета: подсчитывает изображения, ищет битые файлы.
    Также сохраняет полный список валидных изображений в CSV.

    Args:
        data_dir: Путь к директории с изображениями.

    Returns:
        DataFrame со статистикой по каждому художнику.
    """
    print(f" Анализ датасета в директории: {data_dir}")
    artist_images = get_image_paths(data_dir)

    stats = []
    all_images = []

    for artist, images in artist_images.items():
        broken = check_broken_images(images)
        valid_images = [img for img in images if img not in broken]

        if broken:
            print(f" Найдено {len(broken)} битых файлов у художника {artist}")

        stats.append(
            {
                "artist": artist,
                "total_images": len(images),
                "valid_images": len(valid_images),
                "broken_images": len(broken),
            }
        )

        all_images.extend([(artist, img) for img in valid_images])

    df = pd.DataFrame(stats)
    df = df.sort_values("valid_images", ascending=False).reset_index(drop=True)

    images_df = pd.DataFrame(all_images, columns=["artist", "image_path"])

    # Сохраняем общий список изображений
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    images_df.to_csv(
        config.PROCESSED_DATA_DIR / "all_images.csv", index=False, encoding="utf-8-sig"
    )
    print(
        f" Список всех изображений сохранён в {config.PROCESSED_DATA_DIR / 'all_images.csv'}"
    )

    return df


def visualize_class_distribution(df: pd.DataFrame):
    """
    Визуализирует распределение количества картин по художникам.

    Args:
        df: DataFrame со статистикой (результат analyze_dataset).
    """
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].bar(range(len(df)), df["valid_images"])
    axes[0, 0].set_xlabel("Художник (индекс)")
    axes[0, 0].set_ylabel("Количество изображений")
    axes[0, 0].set_title("Распределение изображений по художникам")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(df["valid_images"], bins=20, edgecolor="black")
    axes[0, 1].set_xlabel("Количество изображений")
    axes[0, 1].set_ylabel("Количество художников")
    axes[0, 1].set_title("Гистограмма распределения классов")
    axes[0, 1].grid(alpha=0.3)

    sorted_counts = df["valid_images"].sort_values().values
    cumulative = np.cumsum(sorted_counts) / np.sum(sorted_counts) * 100
    axes[1, 0].plot(range(len(cumulative)), cumulative)
    axes[1, 0].set_xlabel("Художник (отсортировано)")
    axes[1, 0].set_ylabel("Накопленный процент (%)")
    axes[1, 0].set_title("Кумулятивное распределение")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].axhline(y=80, color="r", linestyle="--", label="80%")
    axes[1, 0].legend()

    plt.tight_layout()
    plt.show()


def print_dataset_summary(df: pd.DataFrame):
    """
    Выводит сводную статистику по датасету в консоль.

    Args:
        df: DataFrame со статистикой.
    """
    if df.empty:
        print("Датасет пуст.")
        return

    print(f"Всего художников (классов): {len(df)}")
    print(f"Всего изображений: {df['valid_images'].sum()}")
    print(f"Всего битых файлов: {df['broken_images'].sum()}")
    print(
        f"Среднее количество изображений на художника: {df['valid_images'].mean():.1f}"
    )
    print(f"Медианное количество изображений: {df['valid_images'].median():.1f}")
    print(f"Минимум изображений у: {df['valid_images'].min()}")
    print(f"Максимум изображений у: {df['valid_images'].max()}")
    print(
        f"Дисбаланс классов составляет: {df['valid_images'].max() / max(df['valid_images'].min(), 1):.2f}"
    )

    print("\nТОП-5 художников по количеству работ:")
    for idx, row in df.head(5).iterrows():
        print(f"  {idx + 1}. {row['artist']}: {row['valid_images']} изображений")

    print("\nТОП-5 художников с наименьшим количеством работ:")
    for idx, row in df.tail(5).iterrows():
        print(f"  {idx + 1}. {row['artist']}: {row['valid_images']} изображений")


def stratified_split(
    images_df: pd.DataFrame,
    train_size: float = config.TRAIN_RATIO,
    val_size: float = config.VAL_RATIO,
    test_size: float = config.TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Стратифицированное разделение данных на train/val/test.

    Args:
        images_df: DataFrame с колонками 'artist' и 'image_path'.
        train_size: Доля обучающей выборки.
        val_size: Доля валидационной выборки.
        test_size: Доля тестовой выборки.

    Returns:
        Три DataFrame: train, val, test.
    """
    assert abs(train_size + val_size + test_size - 1.0) < 0.01, (
        "Сумма долей выборок должна быть равна 1.0"
    )

    train_df, temp_df = train_test_split(
        images_df,
        test_size=(val_size + test_size),
        stratify=images_df["artist"],
        random_state=config.RANDOM_SEED,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (val_size + test_size),
        stratify=temp_df["artist"],
        random_state=config.RANDOM_SEED,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Сохраняет разбиения выборки в CSV-файлы.
    """
    config.SPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(
        config.SPLIT_DATA_DIR / "train.csv", index=False, encoding="utf-8-sig"
    )
    val_df.to_csv(config.SPLIT_DATA_DIR / "val.csv", index=False, encoding="utf-8-sig")
    test_df.to_csv(
        config.SPLIT_DATA_DIR / "test.csv", index=False, encoding="utf-8-sig"
    )

    print(
        f" Train: {len(train_df)} изображений -> {config.SPLIT_DATA_DIR / 'train.csv'}"
    )
    print(f" Val: {len(val_df)} изображений -> {config.SPLIT_DATA_DIR / 'val.csv'}")
    print(f" Test: {len(test_df)} изображений -> {config.SPLIT_DATA_DIR / 'test.csv'}")


# =============================================================================
# РАБОТА С МЕТАДАННЫМИ И МАППИНГАМИ
# =============================================================================


def parse_styles(genre_str: str) -> List[str]:
    """
    Парсит строку со стилями, разделёнными запятой.
    """
    if pd.isna(genre_str) or genre_str == "":
        return ["Unknown"]
    styles = [s.strip() for s in str(genre_str).split(",")]
    return [s for s in styles if s]


def extract_era_from_years(years_str: str) -> str:
    """
    Вычисляет эпоху из периода жизни художника (например, "1884 - 1920" -> "19th_century").
    """
    if pd.isna(years_str):
        return "Unknown"
    try:
        birth_year = int(str(years_str).split("-")[0].strip())
    except ValueError:
        return "Unknown"

    if birth_year < 1500:
        return "15th_century"
    elif birth_year < 1600:
        return "16th_century"
    elif birth_year < 1700:
        return "17th_century"
    elif birth_year < 1800:
        return "18th_century"
    elif birth_year < 1900:
        return "19th_century"
    elif birth_year < 2000:
        return "20th_century"
    else:
        return "21st_century"


def load_artist_metadata(csv_path: Path) -> pd.DataFrame:
    """
    Загружает CSV с метаданными и нормализует данные (стили, эпохи).
    """
    if not csv_path.exists():
        print(
            f" Файл метаданных не найден: {csv_path}. Используются значения по умолчанию."
        )
        return pd.DataFrame(columns=["name_clean", "styles", "era"])

    df = pd.read_csv(csv_path)
    df["name_clean"] = df["name"]

    if "genre" in df.columns:
        df["styles"] = df["genre"].apply(parse_styles)
    else:
        df["styles"] = df["name"].apply(lambda x: ["Unknown"])

    if "era" not in df.columns and "years" in df.columns:
        df["era"] = df["years"].apply(extract_era_from_years)
    elif "era" not in df.columns:
        df["era"] = "Unknown"

    return df


def create_artist_mappings(
    train_df: pd.DataFrame,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Создаёт маппинг художник <-> ID на основе тренировочной выборки.
    """
    artists = sorted(
        [str(name).strip().lower() for name in train_df["artist"].unique()]
    )

    artist2id = {artist: idx for idx, artist in enumerate(artists)}
    id2artist = {idx: artist for artist, idx in artist2id.items()}

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.PROCESSED_DATA_DIR / "artist2id.json", "w", encoding="utf-8") as f:
        json.dump(artist2id, f, ensure_ascii=False, indent=2)
    with open(config.PROCESSED_DATA_DIR / "id2artist.json", "w", encoding="utf-8") as f:
        json.dump(id2artist, f, ensure_ascii=False, indent=2)

    print(f"🎨 Создано {len(artists)} классов художников.")
    return artist2id, id2artist


def create_era_and_style_mappings(
    metadata_df: pd.DataFrame,
    artist2id: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Создаёт маппинги для стилей и эпох.
    """
    all_styles = set()
    for styles in metadata_df["styles"]:
        if isinstance(styles, list):
            all_styles.update(styles)
        else:
            all_styles.add("Unknown")

    style2id = {style: idx for idx, style in enumerate(sorted(all_styles))}

    all_eras = metadata_df["era"].dropna().unique().tolist()
    era2id = {era: idx for idx, era in enumerate(sorted(all_eras))}

    with open(config.PROCESSED_DATA_DIR / "style2id.json", "w", encoding="utf-8") as f:
        json.dump(style2id, f, ensure_ascii=False, indent=2)
    with open(config.PROCESSED_DATA_DIR / "era2id.json", "w", encoding="utf-8") as f:
        json.dump(era2id, f, ensure_ascii=False, indent=2)

    print(f" Создано {len(style2id)} классов стилей и {len(era2id)} классов эпох.")
    return style2id, era2id


def create_artist_to_labels_mapping(
    metadata_df: pd.DataFrame,
    artist2id: Dict[str, int],
    style2id: Dict[str, int],
    era2id: Dict[str, int],
) -> Dict[int, Dict[str, any]]:
    """
    Создаёт маппинг artist_id -> {style_ids: [...], era_id: int}.
    """
    mapping = {}
    np.random.seed(config.RANDOM_SEED)

    for _, row in metadata_df.iterrows():
        artist_name = str(row.get("name_clean", "")).strip().lower()

        if artist_name not in artist2id:
            continue

        artist_id = artist2id[artist_name]

        era = str(row.get("era", "Unknown")).strip()
        era_id = era2id.get(era, 0)

        styles = row.get("styles", ["Unknown"])
        if not isinstance(styles, list):
            styles = ["Unknown"]

        style_ids = [style2id.get(s, 0) for s in styles]

        mapping[artist_id] = {"style_ids": style_ids, "era_id": era_id}

    return mapping
