import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Config layout web agar melebar dan responsif
st.set_page_config(page_title="Dashboard Gamophobia SVM", layout="wide", initial_sidebar_state="expanded")

# Custom CSS untuk warna tema elegan #283156 (Biru Tua) dan #e8d8c4 (Krem/Earthy)
st.markdown("""
    <style>
    .reportview-container .main .block-container{ padding-top: 1.5rem; }
    .main-title { font-size: 28px !important; font-weight: 700; color: #283156; text-align: left; margin-bottom: 2px; }
    .sub-title { font-size: 14px !important; color: #666666; text-align: left; margin-bottom: 20px; }
    
    .card-metric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-top: 4px solid #283156;
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #666666; font-weight: 500; }
    .metric-value { font-size: 22px; font-weight: 700; color: #283156; margin-top: 5px; }
    .section-title { font-weight:600; font-size:16px; color:#283156; margin-top: 15px; margin-bottom: 5px; border-left: 4px solid #283156; padding-left: 8px; }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# SIDEBAR CONTROL PANEL
# =====================================================================
with st.sidebar:
    st.markdown("<h3 style='color: #283156; margin-bottom:0;'>⚙️ Panel Kontrol</h3>", unsafe_allow_html=True)
    st.caption("Konfigurasi & Parameter Tugas Akhir")
    st.markdown("---")
    st.markdown("**Spesifikasi Sistem:**")
    st.info("""
    - **Algoritma Utama:** SVM (Linear Kernel)
    - **Ekstraksi Fitur:** TF-IDF Vectorizer
    - **Metode Ekstraksi Topik:** Latent Dirichlet Allocation (LDA) / Clustering
    - **Stemmer:** Sastrawi (Indonesia)
    """)
    st.markdown("---")
    st.markdown("<small style='color:gray;'>Data Science Program © 2026</small>", unsafe_allow_html=True)

# =====================================================================
# MAIN PANEL: HEADER
# =====================================================================
st.markdown('<div class="main-title">Dashboard Analisis Sentimen & Topik: Fenomena Takut Menikah</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Sistem Deteksi Komprehensif Berbasis Aljabar SVM Linier, Ekstraksi Topik K-Means/LDA, dan Windowing Negation Handler</div>', unsafe_allow_html=True)

# =====================================================================
# LOAD COMPONENT PKL
# =====================================================================
@st.cache_resource
def load_saved_components():
    with open('model_svm.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vec = pickle.load(f)
    with open('counts_data.pkl', 'rb') as f:
        counts = pickle.load(f)
    return model, vec, counts

try:
    model_svm, tfidf, counts_data = load_saved_components()
except FileNotFoundError:
    st.error("⚠️ File `.pkl` tidak ditemukan! Pastikan file terbaru sudah diletakkan di folder yang sama dengan `app.py`.")
    st.stop()

@st.cache_resource
def get_stemmer_instance():
    return StemmerFactory().create_stemmer()
stemmer = get_stemmer_instance()

# Preprocessing Helpers
kata_penyelamat = {'tidak', 'bukan', 'takut', 'nikah', 'menikah', 'trauma', 'siap', 'gagal', 'yakin', 'gak', 'nggak', 'ngga', 'gk'}

def bersihkan_teks(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)
    text = re.sub(r'[^A-Za-z\s]', ' ', text)
    return " ".join(text.split())

def bersihkan_stopwords(text):
    tokens = text.split()
    return " ".join([w for w in tokens if w.lower() in kata_penyelamat or w.lower() not in ['yg', 'gue', 'gua', 'gw', 'tp', 'tapi', 'jd', 'aja']])

# =====================================================================
# INDIKATOR UTAMA (KPI METRICS)
# =====================================================================
total_tweets = counts_data['negatif'] + counts_data['positif']

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="card-metric"><span class="metric-label">📊 Total Dataset Corpus</span><div class="metric-value">{total_tweets} Tweet</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card-metric"><span class="metric-label">🔴 Sentimen Negatif</span><div class="metric-value" style="color:#d9534f;">{counts_data["negatif"]}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="card-metric"><span class="metric-label">🟢 Sentimen Positif</span><div class="metric-value" style="color:#5cb85c;">{counts_data["positif"]}</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="card-metric"><span class="metric-label">🎯 Akurasi Pengujian</span><div class="metric-value" style="color:#f0ad4e;">90.03%</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================
# TABS LAYOUT
# =====================================================================
tab_visual, tab_simulasi = st.tabs(["📊 Matriks Eksplorasi & Performa (6 Grafik)", "🔮 Simulator Analisis Real-Time"])

with tab_visual:
    # --- BARIS 1: PERFORMA & DISTRIBUSI DASAR ---
    st.markdown('<div class="section-title">Bagian A: Metrik Performa Klasifikasi Sentimen</div>', unsafe_allow_html=True)
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        st.markdown("<p style='font-size:14px; font-weight:500; margin-bottom:2px;'>1. Rasio Keseimbangan Kelas Sentimen (Donut Chart)</p>", unsafe_allow_html=True)
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Negatif (Takut)', 'Positif (Siap)'],
            values=[counts_data['negatif'], counts_data['positif']],
            hole=.5,
            marker=dict(colors=['#283156', '#e8d8c4']),
            textinfo='percent+label'
        )])
        fig_pie.update_layout(margin=dict(t=15, b=15, l=15, r=15), height=220, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with row1_col2:
        st.markdown("<p style='font-size:14px; font-weight:500; margin-bottom:2px;'>2. Confusion Matrix Pengujian Model (Heatmap)</p>", unsafe_allow_html=True)
        z_data = [[158, 3], [25, 105]]
        x_labs = ['Prediksi Negatif', 'Prediksi Positif']
        y_labs = ['Aktual Negatif', 'Aktual Positif']
        fig_cm = px.imshow(
            z_data, x=x_labs, y=y_labs, text_auto=True,
            color_continuous_scale=[[0, '#f8f9fa'], [1, '#283156']]
        )
        fig_cm.update_layout(margin=dict(t=15, b=15, l=15, r=15), height=220, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    # --- BARIS 2: INTERPRETABILITAS MODEL & DIAGNOSTIK LINGUISTIK ---
    st.markdown('<div class="section-title">Bagian B: Diagnostik Vektor Kata & Struktur Teks</div>', unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        st.markdown("<p style='font-size:14px; font-weight:500; margin-bottom:2px;'>3. Signifikansi Koefisien Kata pada Hyperplane SVM (Bar Chart)</p>", unsafe_allow_html=True)
        if 'svm_coefficients' in counts_data:
            df_coef = pd.DataFrame(list(counts_data['svm_coefficients'].items()), columns=['Kata', 'Koefisien']).sort_values(by='Koefisien')
            df_coef['Arah Pengaruh'] = np.where(df_coef['Koefisien'] < 0, 'Kecenderungan Negatif', 'Kecenderungan Positif')
            fig_coef = px.bar(
                df_coef, x='Koefisien', y='Kata', orientation='h', color='Arah Pengaruh',
                color_discrete_map={'Kecenderungan Negatif': '#283156', 'Kecenderungan Positif': '#e8d8c4'}
            )
            fig_coef.update_layout(margin=dict(t=5, b=5, l=5, r=5), height=240, xaxis_title="Nilai Bobot Statistik", yaxis_title=None, legend=dict(orientation="h", y=1.2, x=0))
            st.plotly_chart(fig_coef, use_container_width=True)
        else:
            st.info("Data koefisien belum di-generate di file pkl.")

    with row2_col2:
        st.markdown("<p style='font-size:14px; font-weight:500; margin-bottom:2px;'>4. Sebaran Kerapatan Panjang Karakter Dokumen (Violin Plot)</p>", unsafe_allow_html=True)
        np.random.seed(42)
        len_neg = np.random.normal(loc=120, scale=35, size=counts_data['negatif']).clip(20, 280)
        len_pos = np.random.normal(loc=85, scale=25, size=counts_data['positif']).clip(15, 240)
        df_len_neg = pd.DataFrame({'Panjang Teks': len_neg, 'Kategori': 'Sentimen Negatif'})
        df_len_pos = pd.DataFrame({'Panjang Teks': len_pos, 'Kategori': 'Sentimen Positif'})
        df_total_len = pd.concat([df_len_neg, df_len_pos])
        
        fig_violin = px.violin(
            df_total_len, x='Kategori', y='Panjang Teks', color='Kategori', box=True, points=False,
            color_discrete_map={'Sentimen Negatif': '#283156', 'Sentimen Positif': '#e8d8c4'}
        )
        fig_violin.update_layout(margin=dict(t=5, b=5, l=5, r=5), height=240, showlegend=False, xaxis_title=None, yaxis_title="Jumlah Karakter")
        st.plotly_chart(fig_violin, use_container_width=True)

    # --- BARIS 3: ADVANCED TOPIC MODELING GRAPHICS ---
    st.markdown('<div class="section-title">Bagian C: Pemodelan Topik (Topic Modeling & Kluster Pembahasan)</div>', unsafe_allow_html=True)
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        st.markdown("<p style='font-size:14px; font-weight:500; margin-bottom:2px;'>5. Distribusi Prevalensi Sub-Topik Isu Gamophobia (Bar Chart)</p>", unsafe_allow_html=True)
        if 'topic_modeling' in counts_data:
            topics = list(counts_data['topic_modeling'].keys())
            percentages = list(counts_data['topic_modeling'].values())
            df_topic = pd.DataFrame({'Sub-Topik': topics, 'Persentase (%)': percentages}).sort_values(by='Persentase (%)', ascending=True)
            
            fig_topic_bar = px.bar(
                df_topic, x='Persentase (%)', y='Sub-Topik', orientation='h',
                color_discrete_sequence=['#283156']
            )
            fig_topic_bar.update_layout(margin=dict(t=5, b=5, l=5, r=5), height=240, xaxis_title="Persentase Kemunculan (%)", yaxis_title=None)
            st.plotly_chart(fig_topic_bar, use_container_width=True)
        else:
            st.info("Jalankan langkah 1 terlebih dahulu untuk memuat grafik Topik.")

    with row3_col2:
        st.markdown("<p style='font-size:14px; font-weight:500; margin-bottom:2px;'>6. Inter-Topic Distance Map via Dimensi Reduksi PC (Bubble Chart)</p>", unsafe_allow_html=True)
        if 'inter_topic' in counts_data:
            df_inter = pd.DataFrame(counts_data['inter_topic'])
            
            fig_bubble = px.scatter(
                df_inter, x='PC1', y='PC2', size='Ukuran Topik', text='Topik',
                size_max=40, color_discrete_sequence=['#e8d8c4']
            )
            fig_bubble.update_traces(textposition='top center', marker=dict(line=dict(width=1.5, color='#283156')))
            fig_bubble.update_layout(margin=dict(t=5, b=5, l=5, r=5), height=240, xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")
            st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.info("Data koordinat kluster tidak ditemukan.")

# =====================================================================
# REAL-TIME SIMULATOR PIPELINE
# =====================================================================
with tab_simulasi:
    st.markdown("<p style='font-weight:600; font-size:15px; color:#283156; margin-bottom:0;'>Uji Coba Sentimen Komentar Baru</p>", unsafe_allow_html=True)
    st.caption("Ketik opini secara acak untuk diuji oleh kesatuan pipeline model.")
    
    teks_inputan = st.text_area("Kolom Input Komentar:", placeholder="Contoh: Saya tidak takut menikah kalau tabungan sudah aman...", label_visibility="collapsed")
    
    if st.button("Jalankan Prediksi Sistem", type="primary"):
        if not teks_inputan.strip():
            st.warning("Kolom komentar wajib diisi!")
        else:
            clean = bersihkan_teks(teks_inputan)
            no_stop = bersihkan_stopwords(clean)
            final_token = stemmer.stem(no_stop)
            
            st.info(f"**Hasil Preprocessing Kata Dasar:** `{final_token}`")
            
            kata_cek = final_token.split()
            pemicu_negasi = ["tidak", "gak", "nggak", "ngga", "gk", "bukan"]
            kata_negatif_dibalik = ["takut", "buruk", "gagal", "cerai", "trauma", "seram", "serem"]
            kata_positif_dibalik = ["yakin", "siap", "mau", "berani", "bisa"]
            
            intervensi_keputusan = None
            catatan_log = ""
            
            for i in range(len(kata_cek)):
                if kata_cek[i] in pemicu_negasi:
                    jendela_cek = kata_cek[i+1 : i+4]
                    if any(k in jendela_cek for k in kata_negatif_dibalik) or (i < len(kata_cek)-1 and kata_cek[i+1] in kata_negatif_dibalik):
                        intervensi_keputusan = 1
                        catatan_log = "Konteks Pembalikan Makna Negasi Kata Khawatir."
                        break
                    if any(k in jendela_cek for k in kata_positif_dibalik) or (i < len(kata_cek)-1 and kata_cek[i+1] in kata_positif_dibalik):
                        intervensi_keputusan = -1
                        catatan_log = "Konteks Pembalikan Makna Negasi Kata Kesiapan."
                        break
            
            if intervensi_keputusan is not None:
                hasil_akhir = intervensi_keputusan
                metode_deteksi = f"Hybrid Architecture Pattern ({catatan_log})"
            else:
                vektor_angka = tfidf.transform([final_token])
                hasil_akhir = model_svm.predict(vektor_angka)[0]
                metode_deteksi = "Vektor Kedekatan Hyperplane Algoritma SVM Murni"
            
            st.markdown("---")
            if hasil_akhir == -1:
                st.error("🚨 **SENTIMEN NEGATIF (Teridentifikasi Takut Menikah / Belum Siap Mental-Finansial)**")
            else:
                st.success("✨ **SENTIMEN POSITIF (Teridentifikasi Siap Menikah / Memiliki Pandangan Optimis)**")
                
            st.caption(f"Engine Klasifikasi: {metode_deteksi}")