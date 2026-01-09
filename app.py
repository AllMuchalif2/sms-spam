import streamlit as st
import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="SMS Spam Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNGSI CLEANING (Wajib ada biar sama dengan training) ---
def bersihkan_teks(teks):
    teks = str(teks).lower()
    teks = re.sub(r'http\S+', '', teks)
    teks = re.sub(r'[^a-z\s]', '', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

# --- LOAD DATA & MODEL ---
@st.cache_data
def load_data():
    # Pastikan nama file CSV sesuai dengan yang ada di GitHub nanti
    df = pd.read_csv('dataset_sms_spam_v1.csv')
    # Bersihkan data untuk keperluan visualisasi
    df['Teks_Bersih'] = df['Teks'].apply(bersihkan_teks)
    label_map = {0: 'Normal', 1: 'Penipuan', 2: 'Promo'}
    df['label_name'] = df['label'].map(label_map)
    return df

@st.cache_resource
def load_model():
    with open('model_sms_spam_nb.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer_sms.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# Load resources
try:
    df = load_data()
    model, vectorizer = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading files: {e}")
    model_loaded = False

# --- TAMPILAN DASHBOARD ---
st.title("🕵️ Dashboard Analisis SMS Spam Indonesia")
st.markdown("### Tugas Akhir Visualisasi Data")
st.markdown("---")

# Sidebar Menu
menu = st.sidebar.radio("Pilih Menu:", ["🤖 Aplikasi Deteksi AI", "📊 Visualisasi Data"])

# --- MENU 1: APLIKASI DETEKSI ---
if menu == "🤖 Aplikasi Deteksi AI":
    st.header("Cek Pesan Mencurigakan")
    st.write("Masukkan teks SMS di bawah ini untuk dideteksi oleh AI.")

    input_sms = st.text_area("Isi Pesan SMS:", height=150, placeholder="Contoh: Selamat anda menang hadiah...")

    if st.button("🔍 Analisis Sekarang"):
        if input_sms and model_loaded:
            # 1. Bersihkan Input
            clean_input = bersihkan_teks(input_sms)
            
            # 2. Transformasi ke Angka
            vec_input = vectorizer.transform([clean_input])
            
            # 3. Prediksi
            prediksi = model.predict(vec_input)[0]
            proba = model.predict_proba(vec_input)[0]
            confidence = proba[prediksi] * 100

            # 4. Tampilkan Hasil
            st.subheader("Hasil Analisis:")
            
            if prediksi == 1: # Penipuan
                st.error(f"🚨 KATEGORI: PENIPUAN (Yakin: {confidence:.1f}%)")
                st.write("Saran: Hati-hati! Jangan klik link atau transfer uang.")
            elif prediksi == 2: # Promo
                st.warning(f"🏷️ KATEGORI: PROMO (Yakin: {confidence:.1f}%)")
                st.write("Saran: Ini hanya iklan produk/operator.")
            else: # Normal
                st.success(f"✅ KATEGORI: NORMAL (Yakin: {confidence:.1f}%)")
                st.write("Saran: Pesan aman.")
            
            # Tampilkan detail probabilitas
            with st.expander("Lihat Detail Matematika AI"):
                st.write(f"Probabilitas Normal: {proba[0]:.4f}")
                st.write(f"Probabilitas Penipuan: {proba[1]:.4f}")
                st.write(f"Probabilitas Promo: {proba[2]:.4f}")

        elif not model_loaded:
            st.error("Model belum dimuat dengan benar.")
        else:
            st.warning("Silakan ketik pesan dulu.")

# --- MENU 2: VISUALISASI DATA ---
elif menu == "📊 Visualisasi Data":
    st.header("Exploratory Data Analysis (EDA)")
    
    # 1. Bar Chart
    st.subheader("1. Sebaran Data per Kategori")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.countplot(x='label_name', data=df, palette='viridis', ax=ax1)
    st.pyplot(fig1)
    st.info("Insight: Data SMS Normal mendominasi, namun data Penipuan cukup untuk dipelajari model.")

    # 2. Histogram Panjang Karakter
    st.subheader("2. Analisis Panjang Karakter SMS")
    df['panjang'] = df['Teks_Bersih'].apply(len)
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x='panjang', hue='label_name', kde=True, palette='Set1', ax=ax2)
    plt.xlim(0, 300)
    st.pyplot(fig2)
    st.info("Insight: SMS Penipuan/Promo cenderung memiliki kalimat yang lebih panjang dibandingkan SMS Normal.")

    # 3. WordCloud
    st.subheader("3. WordCloud (Kata Kunci)")
    kategori_wc = st.selectbox("Pilih Kategori:", ["Penipuan", "Promo", "Normal"])
    
    map_wc_inv = {"Normal": 0, "Penipuan": 1, "Promo": 2}
    subset = df[df['label'] == map_wc_inv[kategori_wc]]
    text_gabung = ' '.join(subset['Teks_Bersih'])
    
    if text_gabung:
        wc = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text_gabung)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.imshow(wc, interpolation='bilinear')
        ax3.axis('off')
        st.pyplot(fig3)
    else:
        st.warning("Tidak ada data untuk kategori ini.")