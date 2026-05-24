
import streamlit as st
from PIL import Image
import numpy as np
from src.inference import get_inference_transforms, load_model, run_inference, GradCAMGenerator, apply_gradcam_overlay, GRAD_CAM_ALPHA, DEVICE

# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(page_title=" Art Attribution AI", layout="wide")
st.title(" Атрибуция произведений искусства с Grad-CAM")
st.caption("Загрузите изображение картины для определения художника, стиля и эпохи с визуализацией внимания модели.")

# Боковая панель
with st.sidebar:
    st.header(" Настройки")
    weights_path = st.text_input("Путь к весам (.pth)", value="./model_final.pth")
    mappings_dir = st.text_input("Папка с маппингами (JSON)", value="./mappings")
    model_type = st.selectbox("Архитектура", ["parallel", "sequential"], index=0)
    load_btn = st.button(" Загрузить модель", type="primary")

# Состояние приложения
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.mappings = None
    st.session_state.transforms = None

if load_btn:
    with st.spinner("Загрузка весов и конфигурации..."):
        try:
            model, id2artist, id2style, id2era = load_model(weights_path, mappings_dir, model_type)
            st.session_state.model = model
            st.session_state.mappings = (id2artist, id2style, id2era)
            st.session_state.transforms = get_inference_transforms()
            st.success("Модель успешно загружена в память!")
        except Exception as e:
            st.error(f"Ошибка загрузки: {e}")
            st.session_state.model = None

# Загрузчик изображений
uploaded_file = st.file_uploader("Загрузите изображение картины", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if st.session_state.model is None:
        st.warning("Сначала загрузите модель через боковую панель.")
    else:
        image = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Оригинал", use_container_width=True)

        with st.spinner("Анализ изображения и генерация Grad-CAM..."):
            try:
                id2a, id2s, id2e = st.session_state.mappings
                results = run_inference(st.session_state.model, img_rgb, st.session_state.transforms, id2a, id2s, id2e)

                st.subheader(" Результаты классификации:")
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric(" Художник", f"{results['artist'][0]}", f"{results['artist'][1]*100:.1f}%")
                mc2.metric(" Стиль", f"{results['style'][0]}", f"{results['style'][1]*100:.1f}%")
                mc3.metric(" Эпоха", f"{results['era'][0]}", f"{results['era'][1]*100:.1f}%")

                # Генерация CAM
                cam_gen = GradCAMGenerator(st.session_state.model)
                transformed = st.session_state.transforms(image=img_rgb)
                input_tensor = transformed['image'].unsqueeze(0).to(DEVICE)
                
                cam = cam_gen.generate(input_tensor)
                overlay = apply_gradcam_overlay(img_rgb, cam, alpha=GRAD_CAM_ALPHA)


                with col2:
                    st.image(overlay, caption=" Grad-CAM (Области внимания модели)", use_container_width=True)

            except Exception as e:
                st.error(f"Ошибка при инференсе: {e}")
                import traceback
                st.code(traceback.format_exc())