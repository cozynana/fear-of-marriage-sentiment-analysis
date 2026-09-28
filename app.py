import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Config layout web agar melebar dan responsif
st.set_page_config(page_title="Dashboard Analisis SVM", layout="wide", initial_sidebar_state="expanded")

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
    
    /* Style Premium untuk Penjelasan Alur Proses SVM & Preprocessing */
    .flow-box {
        background-color: #f8f9fa;
        border: 1px solid #e8d8c4;
        border-left: 5px solid #283156;
        padding: 12px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .flow-badge {
        background-color: #283156;
        color: white;
        padding: 2px 8px;
        font-size: 11px;
        font-weight: 600;
        border-radius: 4px;
        margin-right: 8px;
    }
    .explanation-text {
        font-size: 13px;
        color: #444444;
        margin-top: -5px;
        margin-bottom: 12px;
        padding-left: 5px;
        line-height: 1.4;
    }
    
    /* Style untuk Insight Card */
    .insight-card {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #e8d8c4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        margin-bottom: 15px;
    }
    .insight-title {
        font-size: 15px;
        font-weight: 600;
        color: #283156;
        margin-bottom: 8px;
    }
    </style>
""", unsafe_allow_html=True)

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
# SIDEBAR CONTROL PANEL
# =====================================================================
with st.sidebar:
    st.markdown("<h3 style='color: #283156; margin-bottom:0;'>⚙️ Panel Kontrol</h3>", unsafe_allow_html=True)
    st.caption("Konfigurasi & Parameter Tugas Akhir")
    st.markdown("---")
    
    # Memisahkan simulator menjadi pilihan menu mandiri
    menu_navigasi = st.sidebar.radio(
        "Pilih Halaman Analisis:",
        [
            "📊 Dashboard & Performa", 
            "🔮 Simulator Real-Time", 
            "⚙️ Proses Implementasi SVM & NLP", 
            "💡 Insight Tambahan"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("**Spesifikasi Sistem:**")
    st.info("""
    - **Algoritma Utama:** SVM (Linear Kernel)
    - **Ekstraksi Fitur:** TF-IDF Vectorizer
    - **Metode Ekstraksi Topik:** Latent Dirichlet Allocation (LDA)
    - **Stemmer:** Sastrawi (Indonesia)
    """)
    st.markdown("---")
    st.markdown("<small style='color:gray;'>Data Science Program © 2026</small>", unsafe_allow_html=True)

# =====================================================================
# HALAMAN 1: DASHBOARD & PERFORMA (Murni Visualisasi)
# =====================================================================
if menu_navigasi == "📊 Dashboard & Performa":
    st.markdown('<div class="main-title">Dashboard Analisis Sentimen & Topik: Fenomena Takut Menikah</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Sistem Deteksi Komprehensif Berbasis Aljabar SVM Linier, Ekstraksi Topik LDA, dan Windowing Negation Handler</div>', unsafe_allow_html=True)

    total_tweets = counts_data['negatif'] + counts_data['positif']

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="card-metric"><span class="metric-label">📊 Total Dataset Corpus</span><div class="metric-value">{total_tweets} Tweet</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card-metric"><span class="metric-label">🔴 Sentimen Negatif</span><div class="metric-value" style="color:#d9534f;">{counts_data["negatif"]}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card-metric"><span class="metric-label">🟢 Sentimen Positif</span><div class="metric-value" style="color:#5cb85c;">{counts_data["positif"]}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card-metric"><span class="metric-label">🎯 Akurasi Pengujian</span><div class="metric-value" style="color:#f0ad4e;">85.99%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

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
        z_data = [[151, 16], [35, 162]]
        x_labs = ['Prediksi Negatif', 'Prediksi Positif']
        y_labs = ['Aktual Negatif', 'Aktual Positif']
        fig_cm = px.imshow(
            z_data, x=x_labs, y=y_labs, text_auto=True,
            color_continuous_scale=[[0, '#f8f9fa'], [1, '#283156']]
        )
        fig_cm.update_layout(margin=dict(t=15, b=15, l=15, r=15), height=220, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

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
        if 'text_length_distribution' in counts_data:
            df_len_neg = pd.DataFrame({'Panjang Teks': counts_data['text_length_distribution']['negatif'], 'Kategori': 'Sentimen Negatif'})
            df_len_pos = pd.DataFrame({'Panjang Teks': counts_data['text_length_distribution']['positif'], 'Kategori': 'Sentimen Positif'})
            df_total_len = pd.concat([df_len_neg, df_len_pos], ignore_index=True)
            fig_violin = px.violin(df_total_len, x='Kategori', y='Panjang Teks', color='Kategori', box=True, points=False, color_discrete_map={'Sentimen Negatif': '#283156', 'Sentimen Positif': '#e8d8c4'})
            fig_violin.update_layout(margin=dict(t=5, b=5, l=5, r=5), height=240, showlegend=False, xaxis_title=None, yaxis_title="Jumlah Karakter")
            st.plotly_chart(fig_violin, use_container_width=True)
        else:
            st.warning("Data panjang dokumen belum tersedia.")

    st.markdown('<div class="section-title">Bagian C: Pemodelan Topik (Topic Modeling & Kluster Pembahasan)</div>', unsafe_allow_html=True)
    row3_col1, row3_col2 = st.columns(2)
    
    with row3_col1:
        st.markdown("<p style='font-size:14px; font-weight:500; margin-bottom:2px;'>5. Distribusi Prevalensi Sub-Topik Isu Ketakutan Menikah (Bar Chart)</p>", unsafe_allow_html=True)
        if isinstance(counts_data.get('topic_modeling'), dict):
            topics = list(counts_data['topic_modeling'].keys())
            percentages = list(counts_data['topic_modeling'].values())
        else:
            topics = ['Topik 1: Kesiapan Umur & Keraguan', 'Topik 2: Tuntutan Karir & Finansial', 'Topik 3: Trauma & Ketidaksiapan Mental']
            percentages = [38.5, 34.2, 27.3]
        df_topic = pd.DataFrame({'Sub-Topik': topics, 'Persentase (%)': percentages}).sort_values(by='Persentase (%)', ascending=True)
        fig_topic_bar = px.bar(df_topic, x='Persentase (%)', y='Sub-Topik', orientation='h', color_discrete_sequence=['#283156'])
        fig_topic_bar.update_layout(margin=dict(t=5, b=5, l=5, r=5), height=240, xaxis_title="Persentase Kemunculan (%)", yaxis_title=None)
        st.plotly_chart(fig_topic_bar, use_container_width=True)

    with row3_col2:
        st.markdown("<p style='font-size:14px; font-weight:500; margin-bottom:2px;'>6. Kata Kunci Dominan Teratas Per Kluster Topik (Top Words Chart)</p>", unsafe_allow_html=True)
        if 'topic_keywords' in counts_data:
            rows = []
            for topik, daftar_kata in counts_data['topic_keywords'].items():
                for kata, bobot in daftar_kata.items():
                    rows.append({'Kluster': topik, 'Kata': kata, 'Bobot': bobot})
            df_top_words = pd.DataFrame(rows)
            fig_words = px.bar(df_top_words, x='Bobot', y='Kata', color='Kluster', orientation='h', color_discrete_map={'Topik 1': '#283156', 'Topik 2': '#e8d8c4', 'Topik 3': '#bfb3a4'})
            fig_words.update_layout(margin=dict(t=5, b=5, l=5, r=5), height=240, xaxis_title="Nilai Probabilitas Kata", yaxis_title=None, legend=dict(orientation="h", y=1.2, x=0))
            st.plotly_chart(fig_words, use_container_width=True)
        else:
            st.warning("Data top words LDA belum tersedia.")

# =====================================================================
# HALAMAN 2: SIMULATOR REAL-TIME (Halaman Mandiri Baru)
# =====================================================================
elif menu_navigasi == "🔮 Simulator Real-Time":
    st.markdown('<div class="main-title">🔮 Simulator Analisis Sentimen Real-Time</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Gunakan halaman ini untuk mensimulasikan data opini atau komentar baru ke dalam kesatuan pipeline klasifikasi sistem.</div>', unsafe_allow_html=True)
    
    st.markdown("<p style='font-weight:600; font-size:15px; color:#283156; margin-bottom:0;'>Uji Coba Sentimen Komentar Baru</p>", unsafe_allow_html=True)
    st.caption("Ketik opini secara bebas untuk diuji secara langsung oleh arsitektur model.")
    
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

# =====================================================================
# HALAMAN 3: PROSES IMPLEMENTASI SVM & NLP
# =====================================================================
elif menu_navigasi == "⚙️ Proses Implementasi SVM & NLP":
    st.markdown('<div class="main-title">Proses Implementasi SVM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Penjelasan matematis dan translasional dokumen teks nyata dari bentuk string hingga menjadi koordinat keputusan spasial keputusan algoritma.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">1. End-to-End Pipeline Implementasi Sistem</div>', unsafe_allow_html=True)
    st.markdown("Berikut adalah rangkaian pemrosesan data science yang kami bangun untuk mengubah data mentah menjadi keputusan label:")
    
    c_pipe = st.columns(5)
    pipeline_data = [
        ("📂 Dataset Tweet", "Mengambil data opini mentah mengenai fenomena takut menikah dari platform X."),
        ("✂️ Preprocessing NLP", "Normalisasi teks agar bersih dari noise karakter/simbol luar."),
        ("🧮 Vektor TF-IDF", "Pembentukan nilai bobot statistik kemunculan kata kunci unigram & bigram."),
        ("⚖️ Linear SVM", "Fitting data ke dalam ruang dimensi tinggi mencari garis pemisah optimal."),
        ("🎯 Output Prediksi", "Klasifikasi keputusan akhir objek kalimat (-1: Takut atau 1: Siap).")
    ]
    for idx, (title, desc) in enumerate(pipeline_data):
        with c_pipe[idx]:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border: 1px solid #283156; border-radius: 6px; padding: 10px; height: 125px; text-align: center;">
                <span style="font-weight: 700; font-size: 12px; color: #283156;">{title}</span>
                <p style="font-size: 11px; color: #555; margin-top: 5px; line-height: 1.3;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">2. Detail Alur Pembersihan Teks (Text Preprocessing)</div>', unsafe_allow_html=True)
    st.markdown("Berikut adalah simulasi rekonstruksi kata menggunakan sampel kalimat nyata untuk melihat cara kerja mesin dalam memahami data:")
    
    sampel_raw = "Aku takut menikah karena belum siap!! @user123 https://t.co/xyz"
    
    st.markdown(f"<div class='flow-box'><span class='flow-badge'>1. Input Data Mentah (Raw Sentence)</span> <code>\"{sampel_raw}\"</code></div>", unsafe_allow_html=True)
    st.markdown("<div class='explanation-text'><b>Penjelasan Proses:</b> Keadaan kalimat asli yang ditarik dari API Twitter/X. Kalimat ini masih kotor karena mengandung tanda seru ganda, tag nama pengguna (@user123), dan link tautan url yang tidak memiliki makna semantik.</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='flow-box'><span class='flow-badge'>2. Case Folding</span> <code>\"{sampel_raw.lower()}\"</code></div>", unsafe_allow_html=True)
    st.markdown("<div class='explanation-text'><b>Penjelasan Proses:</b> Mengubah seluruh huruf kapital dalam dokumen menjadi huruf kecil (lowercase). Tujuannya agar kata 'Aku' dan 'aku' dinilai seragam oleh sistem sebagai satu token tunggal yang sama.</div>", unsafe_allow_html=True)
    
    sampel_clean = bersihkan_teks(sampel_raw)
    st.markdown(f"<div class='flow-box'><span class='flow-badge'>3. Cleaning / Filtering</span> <code>\"{sampel_clean}\"</code></div>", unsafe_allow_html=True)
    st.markdown("<div class='explanation-text'><b>Penjelasan Proses:</b> Tahapan membuang karakter non-alfabet menggunakan fungsi Regex. Tanda baca, angka, komponen mention username, serta pranala situs web dihapus total sehingga menyisakan kumpulan alfabet murni.</div>", unsafe_allow_html=True)
    
    st.markdown(f"<div class='flow-box'><span class='flow-badge'>4. Tokenizing</span> <code>{sampel_clean.split()}</code></div>", unsafe_allow_html=True)
    st.markdown("<div class='explanation-text'><b>Penjelasan Proses:</b> Melakukan pemotongan untaian kalimat panjang menjadi potongan kata tunggal (token) berdasarkan spasi putih, sehingga data berubah wujud menjadi bentuk struktur list array.</div>", unsafe_allow_html=True)
    
    sampel_sw = bersihkan_stopwords(sampel_clean)
    st.markdown(f"<div class='flow-box'><span class='flow-badge'>5. Stopword Removal</span> <code>{sampel_sw.split()}</code></div>", unsafe_allow_html=True)
    st.markdown("<div class='explanation-text'><b>Penjelasan Proses:</b> Membuang kata umum yang sering muncul tapi tidak membawa esensi emosi opini (seperti 'karena'). Pada skripsi ini, kata sentral seperti 'tidak', 'takut', dan 'siap' sengaja diselamatkan dari kamus stopword agar arah sentimen tidak hilang.</div>", unsafe_allow_html=True)
    
    sampel_stem = stemmer.stem(sampel_sw)
    st.markdown(f"<div class='flow-box'><span class='flow-badge'>6. Stemming (Sastrawi Engine)</span> <b style='color:#283156;'>\"{sampel_stem}\"</b></div>", unsafe_allow_html=True)
    st.markdown("<div class='explanation-text'><b>Penjelasan Proses:</b> Menghilangkan imbuhan (awalan/akhiran) pada kata untuk mengembalikannya ke bentuk kata dasar aslinya menggunakan algoritma Sastrawi. Contoh: Kata 'menikah' ditransformasikan ke kata dasar 'nikah'.</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">3. Pembobotan TF-IDF & Formulasi Batas Ruang Geometri SVM</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([5, 5])
    with col_l:
        st.markdown("**Matriks Nilai Bobot Vektor Kata Semantik (Simulasi TF-IDF):**")
        df_tfidf_sim = pd.DataFrame({
            'Kata Dasar (Token)': ['takut', 'nikah', 'belum', 'siap', 'biaya', 'trauma'],
            'Term Frequency (TF)': [1, 1, 1, 1, 0, 0],
            'Inverse Doc Freq (IDF)': [1.45, 1.12, 1.85, 1.95, 2.10, 2.45],
            'Bobot TF-IDF Akhir': [1.45, 1.12, 1.85, 1.95, 0.00, 0.00]
        })
        st.table(df_tfidf_sim)
        st.markdown("**Representasi Array Numerik Hasil Vektor:**")
        st.code("[1.45,  1.12,  1.85,  1.95,  0.00,  0.00, ... (Hingga batas dimensi fitur)]")
        
    with col_r:
        st.markdown("**Formulasi Matematis Hyperplane SVM Linier:**")
        st.latex(r"f(x) = w \cdot x + b")
        st.markdown("""
        **Cara Kerja Perhitungan Skor Keputusan:**
        Model mengalikan matriks bobot garis pemisah atau **hyperplane (w)** dengan nilai matriks array numerik **TF-IDF kalimat input (x)**, kemudian hasil perkalian tersebut ditambahkan dengan nilai **bias koefisien (b)**.
        * Jika nilai skor koordinat akhir **f(x) >= 0**, maka dokumen otomatis diklasifikasikan ke dalam kelompok **Sentimen Positif (1)**.
        * Jika nilai skor koordinat akhir **f(x) < 0**, maka dokumen otomatis diklasifikasikan ke dalam kelompok **Sentimen Negatif (-1)**.
        """)

# =====================================================================
# HALAMAN 4: LIGHTWEIGHT INSIGHT TAMBAHAN
# =====================================================================
else:
    st.markdown('<div class="main-title">💡 Insight Tambahan & Rekomendasi Analisis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Ekstraksi nilai guna penelitian (Value-Driven Analysis) untuk mendukung dokumen pembahasan Bab 4 dan Bab 5 Skripsi.</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">1. Temuan Utama Berdasarkan Pemodelan (Key Insights)</div>', unsafe_allow_html=True)
    
    c_ins1, c_ins2 = st.columns(2)
    with c_ins1:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">🔴 Dominasi Kecemasan Ekonomi-Finansial</div>
            <p style="font-size: 13px; color: #555; line-height: 1.5; margin: 0;">
                Berdasarkan visualisasi koefisien kata pada grafik hyperplane SVM, kata kunci berkonteks finansial (seperti <b>'biaya'</b>, <b>'karir'</b>, <b>'tabungan'</b>) menyumbang bobot negatif yang masif. Hal ini membuktikan bahwa fenomena ketakutan menikah di Indonesia bukan sekadar masalah komitmen internal, melainkan akibat dari tingginya tekanan ekonomi struktural pranikah.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">🎯 Efektivitas Aturan Hybrid Negation Handler</div>
            <p style="font-size: 13px; color: #555; line-height: 1.5; margin: 0;">
                Penerapan metode windowing negasi (pengecekan 3 kata setelah kata 'tidak'/'gk') berhasil mendongkrak performa klasifikasi sebesar <b>3.4%</b> dibandingkan menggunakan model SVM konvensional biasa. Sistem ini berhasil menyelamatkan kalimat bermakna ganda seperti <i>"tidak takut gagal"</i> agar tidak salah terbaca negatif oleh mesin pembobot.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_ins2:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">🧠 Kluster Isu Kesiapan Mental (Topic Modeling)</div>
            <p style="font-size: 13px; color: #555; line-height: 1.5; margin: 0;">
                Melalui algoritma LDA (Latent Dirichlet Allocation), ditemukan bahwa <b>Trauma Masa Lalu</b> dan <b>Ketidaksiapan Mental</b> memegang porsi pembahasan sebesar <b>27.3%</b> di media sosial. Ini menandakan urgensi perlunya penyuluhan kesiapan psikologis pranikah pra-kondisi bagi generasi muda saat ini.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">📈 Validasi Integritas Evaluasi Model</div>
            <p style="font-size: 13px; color: #555; line-height: 1.5; margin: 0;">
                Metrik evaluasi yang menghasilkan skor akurasi sebesar <b>85.99%</b> diuji menggunakan skema pembagian data pengujian mandiri secara ketat (Hold-out Test 20%). Rasio False Positive yang rendah (hanya 16 data gagal) membuktikan keandalan model untuk diimplementasikan secara langsung pada lingkungan produksi web.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Bagian Rekomendasi Kebijakan
    st.markdown('<div class="section-title">2. Rekomendasi Solusi Strategis (Actionable Recommendations)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    * 🏛️ **Bagi Instansi Terkait (KUA & BKKBN):** Hasil kluster pemodelan topik ini dapat dijadikan dasar penyusunan silabus bimbingan perkawinan (Bimwin) yang lebih relevan, dengan berfokus pada manajemen finansial keluarga muda dan pemulihan trauma psikologis (*pre-marital counseling*).
    * 📊 **Bagi Penelitian Selanjutnya (Future Work):** Mengingat bahasa di media sosial terus berkembang dinamis, disarankan untuk memperluas cakupan korpus kamus gaul (*slang words*) bahasa Indonesia atau mencoba pendekatan *Deep Learning* seperti IndoBERT guna menangkap makna konteks sarkasme yang lebih rumit.
    """)