import pickle
import streamlit as st
import numpy as np
import base64

# Membaca model dan scaler
try:
    jantung_model = pickle.load(open('Jantung_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Fungsi untuk menambahkan latar belakang
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
            min-height: 100vh;  /* Menambahkan ketinggian minimum agar latar belakang terlihat */
        }}
        </style>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Image file not found at the specified path.")
    except Exception as e:
        st.error(f"An error occurred while setting background: {e}")

# Path ke gambar latar belakang
image_path = 'D:/Studi Independent 7 Mojadiapp/Coba/static/stethoscope-with-copy-space.jpg'
set_background_image(image_path)

# Judul web
st.markdown("<h1 style='text-align: center; color: red;'>🫀 Prediksi Penyakit Jantung 🫀</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Gunakan alat ini untuk memprediksi kemungkinan terkena penyakit jantung berdasarkan data medis.</p>", unsafe_allow_html=True)

# Divider
st.markdown("---")

# Input data
st.markdown("### Input Data")
col1, col2 = st.columns(2)

# Input fields
with col1:
    age = st.text_input('Input age')

with col2:
    anaemia = st.text_input('Input Nilai anaemia')

with col1:
    creatinine_phosphokinase = st.text_input('Input Nilai creatinine_phosphokinase')

with col2:
    diabetes = st.text_input('Input Nilai diabetes')

with col1:
    ejection_fraction = st.text_input('Input Nilai ejection_fraction')

with col2:
    high_blood_pressure = st.text_input('Input Nilai high_blood_pressure')

with col1:
    platelets = st.text_input('Input Nilai platelets')

with col2:
    serum_creatinine = st.text_input('Input Nilai serum_creatinine')

with col1:
    serum_sodium = st.text_input('Input Nilai serum_sodium')

with col2:
    sex = st.text_input('Input Nilai sex')

with col1:
    smoking = st.text_input('Input Nilai smoking')

with col2:
    time = st.text_input('Input Nilai time')
    
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
