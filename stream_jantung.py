import pickle
import streamlit as st
import numpy as np
import base64
import os
from pathlib import Path

# Membaca model dan scaler
try:
    jantung_model = pickle.load(open('Jantung_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Fungsi untuk menambahkan latar belakang menggunakan path relatif
def set_background_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_image});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            min-height: 100vh;
        }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Image file not found at the specified path.")
    except Exception as e:
        st.error(f"An error occurred while setting background: {e}")

# Path ke gambar latar belakang relatif terhadap lokasi skrip
current_dir = Path(__file__).parent
image_path = current_dir / 'static' / 'stethoscope-with-copy-space.jpg'
set_background_image(str(image_path))

# Judul web
st.markdown("<h1 style='text-align: center; color: red;'>🫀 Prediksi Penyakit Jantung 🫀</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Gunakan alat ini untuk memprediksi kemungkinan terkena penyakit jantung berdasarkan data medis.</p>", unsafe_allow_html=True)

# Divider
st.markdown("---")

# Input data
st.markdown("### Input Data")
col1, col2 = st.columns(2)

# Input fields dengan number_input
with col1:
    age = st.number_input('Input age', min_value=0, max_value=120, value=50, help="Masukkan usia pasien dalam tahun")
    
with col2:
    anaemia = st.number_input('Input Nilai anaemia', min_value=0, max_value=1, value=0, help="1 jika pasien menderita anemia, 0 jika tidak")
    
with col1:
    creatinine_phosphokinase = st.number_input('Input Nilai creatinine_phosphokinase', min_value=0, value=100, help="Level CPK")
    
with col2:
    diabetes = st.number_input('Input Nilai diabetes', min_value=0, max_value=1, value=0, help="1 jika pasien menderita diabetes, 0 jika tidak")
    
with col1:
    ejection_fraction = st.number_input('Input Nilai ejection_fraction', min_value=0, max_value=100, value=50, help="Persentase ejection fraction")
    
with col2:
    high_blood_pressure = st.number_input('Input Nilai high_blood_pressure', min_value=0, max_value=1, value=0, help="1 jika pasien memiliki tekanan darah tinggi, 0 jika tidak")
    
with col1:
    platelets = st.number_input('Input Nilai platelets', min_value=0, value=300000, help="Jumlah trombosit")
    
with col2:
    serum_creatinine = st.number_input('Input Nilai serum_creatinine', min_value=0.0, value=1.0, help="Level serum creatinine")
    
with col1:
    serum_sodium = st.number_input('Input Nilai serum_sodium', min_value=0, value=135, help="Level serum sodium")
    
with col2:
    sex = st.number_input('Input Nilai sex', min_value=0, max_value=1, value=0, help="1 untuk pria, 0 untuk wanita")
    
with col1:
    smoking = st.number_input('Input Nilai smoking', min_value=0, max_value=1, value=0, help="1 jika pasien merokok, 0 jika tidak")
    
with col2:
    time = st.number_input('Input Nilai time', min_value=0, value=100, help="Waktu follow-up (dalam hari)")

# Divider
st.markdown("---")

# Prediksi
if st.button('🔍 Prediksi Penyakit Jantung'):
    try:
        input_data = [
            float(age), 
            float(anaemia), 
            float(creatinine_phosphokinase), 
            float(diabetes), 
            float(ejection_fraction), 
            float(high_blood_pressure),
            float(platelets), 
            float(serum_creatinine), 
            float(serum_sodium), 
            float(sex), 
            float(smoking), 
            float(time)
        ]

        # Normalisasi data menggunakan scaler
        input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))

        # Prediksi menggunakan model
        jantung_prediction = jantung_model.predict(input_data_scaled)

        if jantung_prediction[0] == 1:
            st.success('✅ Pasien terkena penyakit jantung.')
        else:
            st.success('❌ Pasien tidak terkena penyakit jantung.')

    except ValueError as e:
        st.error(f"Input tidak valid: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
