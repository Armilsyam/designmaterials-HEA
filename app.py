# Simpan kode ini ke dalam file bernama: app.py
# Pastikan file model 'HEA_PINN_Evaluator.keras' berada di folder yang sama
# Jalankan di terminal dengan perintah: streamlit run app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==============================================================================
# 1. KONFIGURASI HALAMAN DAN MEMUAT MODEL
# ==============================================================================
st.set_page_config(page_title="Simulasi HEA Dual-Pipeline AI", layout="wide")

@st.cache_resource
def load_ai_model():
    # Memuat model PINN yang sudah kita simpan di fase sebelumnya
    return load_model('HEA_PINN_Evaluator.keras', compile=False)

try:
    model = load_ai_model()
    model_loaded = True
except:
    model_loaded = False

# ==============================================================================
# 2. ANTARMUKA PENGGUNA (UI)
# ==============================================================================
st.title("🔬 Simulator High-Entropy Alloys (Fe-Cr-Ni-Mo-Al)")
st.markdown("""
Aplikasi ini merupakan implementasi *Dual-Pipeline Hybrid AI* untuk memprediksi 
ketahanan mekanik dan stabilitas termodinamika paduan pelindung reaktor nuklir Gen-IV.
""")

if not model_loaded:
    st.error("Model AI tidak ditemukan! Pastikan 'HEA_PINN_Evaluator.keras' ada di direktori Anda.")
else:
    # Membuat 2 kolom layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Konfigurasi Fraksi Elemen")
        st.markdown("Atur komposisi logam (sistem akan otomatis menormalisasi ke 100%):")
        
        # Slider input dari pengguna
        fe = st.slider("Besi (Fe)", 0.0, 100.0, 48.0)
        cr = st.slider("Kromium (Cr)", 0.0, 100.0, 20.0)
        ni = st.slider("Nikel (Ni)", 0.0, 100.0, 10.0)
        mo = st.slider("Molybdenum (Mo)", 0.0, 100.0, 2.0)
        al = st.slider("Aluminium (Al)", 0.0, 100.0, 20.0)
        
        # Normalisasi otomatis ke 1.0 (100%)
        total = fe + cr + ni + mo + al
        fe_n, cr_n, ni_n, mo_n, al_n = fe/total, cr/total, ni/total, mo/total, al/total

        st.info(f"Total Aktual: Fe({fe_n*100:.1f}%) | Cr({cr_n*100:.1f}%) | Ni({ni_n*100:.1f}%) | Mo({mo_n*100:.1f}%) | Al({al_n*100:.1f}%)")

    with col2:
        st.subheader("Hasil Analisis In-Silico (Prediksi PINN)")
        
        if st.button("Jalankan Prediksi AI", type="primary"):
            # Format input ke array numpy
            input_data = np.array([[fe_n, cr_n, ni_n, mo_n, al_n]])
            
            with st.spinner("AI sedang mengevaluasi hukum termodinamika..."):
                # Prediksi
                hasil = model.predict(input_data)
                pred_yield = hasil[0][0][0]
                pred_omega = hasil[1][0][0]
                pred_delta = hasil[2][0][0]
                
                # Logika Klasifikasi
                fasa = "BCC (Sangat Kuat)" if pred_yield >= 1000 else "FCC (Ulet / Lemah)"
                status_termo = "Stabil" if pred_omega >= 1.1 else "Rawan Fasa Rapuh (Tidak Stabil)"
                
                # Tampilan Hasil metrik
                st.metric(label="Prediksi Kekuatan Luluh (Yield Strength)", value=f"{pred_yield:.2f} MPa")
                
                # Tampilan Fasa
                if pred_yield >= 1000:
                    st.success(f"Fasa Matriks: {fasa}")
                else:
                    st.warning(f"Fasa Matriks: {fasa}")
                
                # Tampilan Kepatuhan Fisika
                st.markdown("### Evaluasi Hukum Alam")
                if pred_omega >= 1.1:
                    st.success(f"Parameter Omega: {pred_omega:.2f} -> {status_termo}")
                else:
                    st.error(f"Parameter Omega: {pred_omega:.2f} -> {status_termo}")
                    
                st.info(f"Distorsi Kisi (Delta): {pred_delta:.2f}% (Batas aman < 6.6%)")
