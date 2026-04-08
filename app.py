import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.preprocessing import detrend_light_curve
from src.features import compute_fft
import numpy as np

# Настройка заголовка страницы
st.title("🌌 Exoplanet Detection System")
st.write("Upload a light curve CSV to check for exoplanets.")

# 1. Загрузка модели
@st.cache_resource # Чтобы не грузить модель каждый раз заново
def load_model():
    return joblib.load('models/rf_baseline.pkl')

model = load_model()

# 2. Загрузчик файлов (File Uploader)
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Читаем данные
    data = pd.read_csv(uploaded_file)
    flux = data.iloc[0].values # Берем первую строку для примера
    
    st.subheader("Raw Light Curve Visualization")
    fig, ax = plt.subplots()
    ax.plot(flux, color='gray', alpha=0.5)
    st.pyplot(fig)

    # 3. Препроцессинг (Тот самый пайплайн!)
    st.info("Processing signal: Detrending + FFT...")
    clean_flux = detrend_light_curve(flux)
    
    # Предположим, dt у нас константа из датасета
    dt = 0.0204 
    freqs, power = compute_fft(clean_flux, dt)
    
    # 4. Предсказание (Inference)
    # Нам нужно подготовить вектор признаков точно так же, как при обучении
    # (Здесь должна быть логика объединения стат-фич и FFT)
    features = power[:782].reshape(1, -1) 
    
    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    # 5. Вывод результата (User Experience)
    if prediction[0] == 1:
        st.success(f"🎯 EXOPLANET DETECTED! Confidence: {probability:.2%}")
    else:
        st.error(f"🚫 No Exoplanet Found. Confidence: {1-probability:.2%}")