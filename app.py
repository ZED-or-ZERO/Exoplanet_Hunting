import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import time
import os

# Импорт наших кастомных модулей из папки src
from src.preprocessing import detrend_light_curve
from src.features import compute_fft

# Для 1D-CNN (нужно импортировать архитектуру, чтобы загрузить веса)
from src.model import ExoplanetCNN 
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Настройка страницы (Page Config)
# ==========================================
st.set_page_config(page_title="Exoplanet Batch Analyzer", layout="wide")
st.title("🌌 Exoplanet Batch Analyzer")
st.markdown("Upload raw datasets to process and detect exoplanets using Machine Learning.")

# ==========================================
# 2. Сайдбар: Загрузка файлов и выбор модели
# ==========================================
with st.sidebar:
    st.header("1. Upload Datasets")
    flux_file = st.file_uploader("Upload Raw Flux (CSV)", type="csv")
    time_file = st.file_uploader("Upload Time (CSV)", type="csv")
    features_file = st.file_uploader("Upload Features (CSV)", type="csv")
    
    st.header("2. Model Selection")
    model_choice = st.radio(
        "Choose Inference Engine:",
        ("Random Forest (Classic ML)", "1D-CNN (Deep Learning)")
    )
    
    start_button = st.button("🚀 Start Batch Processing", use_container_width=True)

# ==========================================
# 3. Кэшированные функции (Performance Optimization)
# ==========================================
@st.cache_resource
def load_rf_model():
    return joblib.load('models/rf_baseline.pkl')

@st.cache_resource
def load_cnn_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExoplanetCNN().to(device)
    # Предполагается, что ты сохранил веса через torch.save(model.state_dict(), 'models/cnn_weights.pth')
    model.load_state_dict(torch.load('models/cnn_weights.pth', map_location=device))
    model.eval()
    return model, device

# ==========================================
# 4. Основная логика обработки (Data Pipeline)
# ==========================================
if start_button:
    if not (flux_file and time_file and features_file):
        st.error("Please upload all three CSV files to begin processing.")
    else:
        try:
            # Чтение файлов (Data Loading)
            with st.spinner("Loading 1GB datasets into memory..."):
                flux_df = pd.read_csv(flux_file)
                time_df = pd.read_csv(time_file)
                features_df = pd.read_csv(features_file)
            
            st.success(f"Datasets loaded! Total stars to process: {len(flux_df)}")
            
            # Подготовка прогресс-бара
            progress_text = "Applying Detrending and FFT..."
            my_bar = st.progress(0, text=progress_text)
            
            # Массивы для новых данных
            processed_flux_list = []
            fft_features_list = []
            
            # Предполагаем, что шаг времени константный для всего датасета
            dt = time_df.iloc[0, 1] - time_df.iloc[0, 0] 
            
            # Симуляция пакетной обработки (Batch Processing)
            total_rows = len(flux_df)
            
            # Очистка от 'id' и 'class', если они есть
            raw_flux_values = flux_df.drop(columns=['id', 'class'], errors='ignore').values
            
            # Процесс очистки (Signal Processing)
            for i in range(total_rows):
                raw_signal = raw_flux_values[i]
                
                # 1. Detrending
                clean_signal = detrend_light_curve(raw_signal)
                processed_flux_list.append(clean_signal)
                
                # 2. FFT
                _, power = compute_fft(clean_signal, dt)
                fft_features_list.append(power[:500]) # Берем 500 фич
                
                # Обновление UI каждые 100 строк, чтобы не тормозить браузер
                if i % 100 == 0:
                    my_bar.progress(i / total_rows, text=f"Processing signal {i}/{total_rows}")
            
            my_bar.progress(1.0, text="Preprocessing complete!")
            
            # Сохранение чистого датасета (Exporting clean_data.csv)
            with st.spinner("Saving processed data to clean_data.csv..."):
                clean_flux_df = pd.DataFrame(processed_flux_list)
                clean_flux_df.to_csv("clean_data.csv", index=False)
                st.info("💾 Processed signals saved locally as `clean_data.csv`.")

            # ==========================================
            # 5. Машинное обучение (Inference)
            # ==========================================
            st.subheader(f"Analyzing with {model_choice}...")
            
            predictions = []
            
            if model_choice == "Random Forest (Classic ML)":
                rf_model = load_rf_model()
                
                # Подготовка табличных признаков (Feature Fusion)
                feature_cols = [f'freq_{i+1}' for i in range(500)]
                fft_df = pd.DataFrame(fft_features_list, columns=feature_cols)
                
                # Избавляемся от id и class в features_df для слияния
                stats_df = features_df.drop(columns=['id', 'class'], errors='ignore')

                # 2. ЖЕСТКИЙ ФИЛЬТР: Удаляем ВСЕ колонки, в названии которых есть слово "Unnamed"
                stats_df = stats_df.loc[:, ~stats_df.columns.str.contains('^Unnamed')]
                
                # Конкатенация по колонкам (соединяем стат-фичи и FFT)
                final_X = pd.concat([stats_df, fft_df], axis=1)
                
                with st.spinner("Running Random Forest Inference..."):
                    predictions = rf_model.predict(final_X)
                    
            elif model_choice == "1D-CNN (Deep Learning)":
                cnn_model, device = load_cnn_model()
                
                with st.spinner("Scaling and running Deep Learning Inference..."):
                    # Для CNN нужен StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(np.array(processed_flux_list))
                    
                    # Конвертация в тензор [Batch, Channels, Length]
                    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1).to(device)
                    
                    # Inference
                    with torch.no_grad():
                        outputs = cnn_model(X_tensor)
                        # Превращаем вероятности в классы (0 или 1)
                        predictions = (outputs >= 0.5).float().cpu().numpy().flatten()
            
            # ==========================================
            # 6. Результаты (Dashboard Output)
            # ==========================================
            total_planets = int(np.sum(predictions))
            total_scanned = len(predictions)
            
            st.markdown("---")
            st.header("🎯 Final Results")
            
            col1, col2 = st.columns(2)
            col1.metric("Total Stars Scanned", total_scanned)
            col2.metric("🪐 Exoplanets Detected", total_planets, delta="Candidates Found", delta_color="normal")
            
            if total_planets > 0:
                st.balloons()
                
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")