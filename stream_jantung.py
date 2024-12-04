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

# Fungsi untuk menambahkan latar belakang menggunakan URL gambar dari GitHub
def set_background_image(image_url):
    try:
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url({image_url});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            min-height: 100vh;
        }}
        </style>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred while setting background: {e}")

# URL gambar di GitHub
image_url = 'https://raw.githubusercontent.com/Salmangarifatul/Prediksi_Jantung5/main/stethoscope-with-copy-space.jpg'
set_background_image(image_url)

# Menambahkan CSS untuk teks hitam, kolom input putih, dan judul hitam
st.markdown("""
    <style>
        .stApp {
            color: black !important;  /* Mengatur warna teks di seluruh aplikasi menjadi hitam */
        }
        .css-1v3fvcr {  /* CSS untuk kolom input */
            background-color: white !important;
            color: black !important;
        }
        .stTextInput input {
            background-color: white !important;
            color: black !important;
        }
        h1, h2, h3 {  /* Mengatur warna font judul menjadi hitam */
            color: black !important;
        }
        /* Menargetkan teks input di dalam kolom untuk menjadi hitam */
        .stTextInput label {
            color: black !important;  /* Mengatur warna label input menjadi hitam */
        }
        /* CSS untuk kolom prediksi agar latar belakangnya putih */
        .stButton > button {
            background-color: white !important;
            color: black !important;
            border: 2px solid #ddd !important;
        }
        /* CSS untuk hasil prediksi agar teks berwarna hitam */
        .stSuccess, .stError {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)

# Judul web
st.markdown("<h1 style='text-align: center; color: red;'>🫀 Prediksi Penyakit Jantung 🫀</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Gunakan alat ini untuk memprediksi kemungkinan terkena penyakit jantung berdasarkan data medis.</p>", unsafe_allow_html=True)

# Divider
st.markdown("---")

# Input data
st.markdown("### Input Data")
col1, col2 = st.columns(2)

# Input fields dengan number_input
with col1 :
    age = st.text_input('Input age')

with col2 :
    anaemia = st.text_input('Input Nilai anaemia')

with col1 :
    creatinine_phosphokinase = st.text_input('Input Nilai creatinine_phosphokinase')

with col2 :
    diabetes = st.text_input('Input Nilai diabetes')

with col1 :
    ejection_fraction = st.text_input('Input Nilai ejection_fraction')

with col2 :
    high_blood_pressure = st.text_input('Input Nilai high_blood_pressure')

with col1 :
    platelets = st.text_input('Input Nilai platelets')

with col2 :
    serum_creatinine = st.text_input('Input Nilai serum_creatinine')

with col1 :
    serum_sodium = st.text_input('Input Nilai serum_sodium')

with col2 :
    sex = st.text_input('Input Nilai sex')

with col1 :
    smoking = st.text_input('Input Nilai smoking')

with col2 :
    time = st.text_input('Input Nilai time')
    
# Divider
st.markdown("---")

# code untuk prediksi
jantung_diagnosis = ''

# membuat tombol untuk prediksi
st.markdown("<h3 style='text-align: center;'>Hasil Prediksi</h3>", unsafe_allow_html=True)
if st.button('🔍 Prediksi Penyakit Jantung'):
    try:
        # Konversi input menjadi float dan numerik
        input_data = [
            float(age), 
            int(anaemia.split(':')[0]), 
            float(creatinine_phosphokinase), 
            int(diabetes.split(':')[0]), 
            float(ejection_fraction), 
            int(high_blood_pressure.split(':')[0]),
            float(platelets), 
            float(serum_creatinine), 
            float(serum_sodium), 
            int(sex.split(':')[0]), 
            int(smoking.split(':')[0]), 
            float(time)
        ]

        # Konversi ke array 2D
        input_data = np.array(input_data).reshape(1, -1)

        # Normalisasi data menggunakan scaler
        input_data_scaled = scaler.transform(input_data)

        # Prediksi menggunakan model
        jantung_prediction = jantung_model.predict(input_data_scaled)

        if jantung_prediction[0] == 1:
            st.success('✅ Pasien terkena penyakit jantung.')
        else:
            st.success('❌ Pasien tidak terkena penyakit jantung.')

        # Tampilkan hasil
        st.success(jantung_diagnosis)

    except ValueError as e:
        st.error(f"Input tidak valid: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
