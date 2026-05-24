import torch
from pathlib import Path

# =============================================================================
# БАЗОВЫЕ ПУТЫ
# =============================================================================
# При локальном запуске замените KAGGLE_WORKING / KAGGLE_INPUT на абсолютные пути
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

DATA_DIR_OUT = KAGGLE_INPUT / "datasets/antineutrinoneutrino/my-lovely-dataset"
RAW_DATA_DIR = DATA_DIR_OUT / "raw"
PROCESSED_DATA_DIR = DATA_DIR_OUT / "mappings"
SPLIT_DATA_DIR = DATA_DIR_OUT / "splitted_data"

RESULTS_DIR = KAGGLE_WORKING / "results"
LOG_DIR = RESULTS_DIR / "logs"
MODELS_DIR = KAGGLE_WORKING / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints_model"
EXPORT_DIR = MODELS_DIR / "export"
ONNX_DIR = EXPORT_DIR / "onnx"
TORCHSCRIPT_DIR = EXPORT_DIR / "torchscript"
PACKAGE_DIR = EXPORT_DIR / "package"

ARTIST_METADATA_CSV = (
    KAGGLE_INPUT / "datasets/ikarus777/best-artworks-of-all-time/artists.csv"
)

# Визуализация
VISUALIZATION_DIR = RESULTS_DIR / "visualization"
GRAD_CAM_DIR = VISUALIZATION_DIR / "grad_cam"
ATTENTION_DIR = VISUALIZATION_DIR / "attention_maps"
COMBINED_DIR = VISUALIZATION_DIR / "combined"
COMPARISON_DIR = VISUALIZATION_DIR / "architecture_comparison"

# =============================================================================
# РАЗДЕЛЕНИЕ ВЫБОРКИ
# =============================================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# =============================================================================
# ПРЕДОБРАБОТКА И ЗАГРУЗКА ДАННЫХ
# =============================================================================
IMG_SIZE = 224
IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
TEST_BATCH_SIZE = 32
TEST_NUM_WORKERS = 4

# =============================================================================
# АРХИТЕКТУРЫ МОДЕЛЕЙ
# =============================================================================
HEAD_DROPOUT = 0.0
CNN_ARCH = "efficientnet_b0"
TRANSFORMER_ARCH = "swin_tiny_patch4_window7_224"
EMBEDDING_DIM = 96
HYBRID_TYPES = ["sequential", "parallel"]
HYBRID_TYPE = HYBRID_TYPES[1]  # "parallel" по умолчанию

# =============================================================================
# ОБУЧЕНИЕ И ОПТИМИЗАЦИЯ
# =============================================================================
TOTAL_EPOCHS = 50
LR_WARMUP_EPOCHS = 5
OPTIMIZER = "adam"  # "adam", "adamw", "sgd"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
LR_SCHEDULER = "cosine"  # 'cosine', 'step', 'plateau'
LABEL_SMOOTHING = 0.1
TASK_WEIGHTS = {"artist": 6.0, "style": 1.0, "era": 1.0}
USE_CLASS_WEIGHTS = True
USE_AMP = True  # Автоматическая смешанная точность

# =============================================================================
# РАННЯЯ ОСТАНОВКА И ЛОГИРОВАНИЕ
# =============================================================================
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001

TENSORBOARD_ENABLED = True
CSV_LOG_ENABLED = True

SAVE_BEST_MODEL = True
SAVE_LAST_MODEL = True
SAVE_EVERY_N_EPOCHS = 5

# =============================================================================
# ОЦЕНКА И СРАВНЕНИЕ АРХИТЕКТУР
# =============================================================================
COMPARE_ARCHITECTURES = True
ARCHITECTURES_TO_COMPARE = ["sequential", "parallel"]

# =============================================================================
# ВИЗУАЛИЗАЦИЯ И ИНТЕРПРЕТАЦИЯ (Grad-CAM, Attention)
# =============================================================================
GRAD_CAM_ALPHA = 0.4
GRAD_CAM_TARGET_LAYER = None
GRAD_CAM_UPSAMPLE_METHOD = "bilinear"
CAM_COLORMAP = "jet"

ATTENTION_AGGREGATION = "mean"
ATTENTION_NORMALIZE = True
ATTENTION_ALPHA = 0.5
ATTENTION_COLORMAP = "viridis"

NUM_IMAGES_TO_VISUALIZE = 20
NUM_CORRECT_IMAGES = 10
NUM_INCORRECT_IMAGES = 10

# =============================================================================
# ПАРАМЕТРЫ СОХРАНЕНИЯ И ЭКСПОРТА
# =============================================================================
DPI_FOR_PLOTS = 300
FIGSIZE = (12, 10)
SAVE_FORMATS = ["png", "pdf", "csv", "json", "svg"]

EXPORT_ONNX = True
EXPORT_TORCHSCRIPT = True
EXPORT_PACKAGE = True

ONNX_OPSET_VERSION = 13
ONNX_DYNAMIC_AXES = True
ONNX_OPTIMIZE = True

TORCHSCRIPT_TRACE = True
TORCHSCRIPT_OPTIMIZE = True

INCLUDE_CONFIG = True
INCLUDE_TRANSFORMS = True
INCLUDE_CLASS_MAPPINGS = True
INCLUDE_INFERENCE_SCRIPT = True
CREATE_REQUIREMENTS = True

# =============================================================================
# МЕТАДАННЫЕ МОДЕЛИ И ВЫЧИСЛИТЕЛЬНОЕ УСТРОЙСТВО
# =============================================================================
MODEL_NAME = "ArtAttributionHybrid"
MODEL_VERSION = "1.0.0"
MODEL_AUTHOR = "Student Name"
MODEL_DESCRIPTION = "Гибридная нейросетевая модель для атрибуции произведений живописи"
MODEL_LICENSE = "MIT"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXPORT_DEVICE = "cpu"  # 'cpu' для совместимости, 'cuda' для производительности
