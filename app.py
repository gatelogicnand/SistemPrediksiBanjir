import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN (Wajib di baris pertama)
# ==========================================
st.set_page_config(
    page_title="Flood Risk Lhokseumawe",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS CUSTOM (Untuk Estetika Profesional)
# ==========================================
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #007bff;
        color: white;
    }
    
    /* Styling Container Metric */
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }

    /* PERBAIKAN: Memaksa warna teks Label (judul kecil) menjadi gelap */
    [data-testid="stMetricLabel"] {
        color: #444444 !important;
    }

    /* PERBAIKAN: Memaksa warna teks Value (angka besar) menjadi hitam */
    [data-testid="stMetricValue"] {
        color: #000000 !important;
    }

    h1, h2, h3 {
        color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 3. FUNGSI LOAD MODEL
# ==========================================
@st.cache_resource
def load_models():
    try:
        model = joblib.load('kmeans_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return model, scaler
    except FileNotFoundError:
        st.error("File model tidak ditemukan. Pastikan 'kmeans_model.joblib' dan 'scaler.joblib' ada di folder yang sama.")
        return None, None

model, scaler = load_models()

# ==========================================
# 4. LOGIKA INTERPRETASI KLASTER
# ==========================================
def get_cluster_info(model, scaler):
    """
    Mengidentifikasi label klaster (0, 1, 2) menjadi tingkat risiko
    berdasarkan karakteristik centroid (Elevasi & Curah Hujan).
    """
    centers_scaled = model.cluster_centers_
    # Inverse transform untuk melihat nilai asli (mm dan mdpl)
    centers_original = scaler.inverse_transform(centers_scaled)
    
    # Buat DataFrame Centroid
    df_centers = pd.DataFrame(centers_original, columns=['Curah_Hujan', 'Elevasi'])
    df_centers['Cluster_Label'] = range(len(df_centers))
    
    # Logika Penentuan Risiko:
    # Elevasi rendah & Hujan tinggi = Sangat Rawan
    # Elevasi tinggi = Aman
    # Kita urutkan berdasarkan Elevasi (Ascending: Rendah ke Tinggi)
    df_sorted = df_centers.sort_values(by='Elevasi').reset_index(drop=True)
    
    # Mapping dinamis berdasarkan urutan elevasi
    # Klaster dengan elevasi terendah -> "Sangat Rawan"
    # Klaster tengah -> "Rawan" / "Siaga"
    # Klaster elevasi tertinggi -> "Aman"
    
    risk_mapping = {}
    colors = {}
    
    # Asumsi 3 klaster (jika jumlah klaster berbeda, logika ini menyesuaikan urutan)
    labels = ["Sangat Rawan (Bahaya)", "Waspada (Siaga)", "Aman"]
    color_codes = ["#FF4B4B", "#FFA500", "#28A745"] # Merah, Oranye, Hijau
    
    for i in range(len(df_sorted)):
        original_label = df_sorted.loc[i, 'Cluster_Label']
        if i < len(labels):
            risk_mapping[original_label] = labels[i]
            colors[original_label] = color_codes[i]
        else:
            risk_mapping[original_label] = f"Level {i}"
            colors[original_label] = "#808080"
            
    return df_centers, risk_mapping, colors

if model is not None:
    df_centers, risk_map, color_map = get_cluster_info(model, scaler)

# ==========================================
# 5. SIDEBAR (INPUT USER)
# ==========================================
with st.sidebar:
    st.title("Sistem Prediksi Banjir")
    st.markdown("Dashboard ini menggunakan **Machine Learning (K-Means)** untuk mengelompokkan tingkat kerawanan banjir di Kota Lhokseumawe.")
    
    st.markdown("---")
    st.header("🔧 Input Parameter")
    
    input_kecamatan = st.selectbox("Pilih Kecamatan", 
                                   ["Banda Sakti", "Blang Mangat", "Muara Dua", "Muara Satu"])
    
    input_gampong = st.text_input("Nama Desa (Opsional)", "Contoh: Hagu Teungoh")
    
    # Slider dengan range yang masuk akal untuk Lhokseumawe
    val_hujan = st.slider("🌧️ Curah Hujan (mm/bulan)", min_value=0.0, max_value=600.0, value=250.0, step=0.1)
    val_elevasi = st.slider("⛰️ Elevasi (mdpl)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    
    btn_predict = st.button("Analisa Tingkat Kerawanan")

# ==========================================
# 6. HALAMAN UTAMA
# ==========================================
st.title("🌊 Dashboard Kerawanan Banjir Kota Lhokseumawe")
st.markdown(f"Selamat datang di sistem pendukung keputusan mitigasi bencana. Data input: **{input_kecamatan}**.")

# Tab Layout
tab1, tab2 = st.tabs(["📊 Analisa & Prediksi", "ℹ️ Informasi Klaster"])

with tab1:
    if model is not None:
        # --- PROSES PREDIKSI ---
        # 1. Scale data input user
        input_data = np.array([[val_hujan, val_elevasi]])
        input_scaled = scaler.transform(input_data)
        
        # 2. Prediksi Klaster
        prediction = model.predict(input_scaled)[0]
        result_text = risk_map[prediction]
        result_color = color_map[prediction]
        
        # --- TAMPILAN METRICS ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Curah Hujan Input", value=f"{val_hujan} mm")
        with col2:
            st.metric(label="Elevasi Input", value=f"{val_elevasi} mdpl")
        with col3:
            st.markdown(f"""
            <div style="background-color: {result_color}; color: white; padding: 10px; border-radius: 5px; text-align: center;">
                <h4 style="margin:0; color:white;">Status: {result_text}</h4>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # --- VISUALISASI PLOTLY ---
        st.subheader("📍 Posisi Data dalam Klaster")
        
        # 1. Buat data dummy background untuk visualisasi area klaster
        # (Karena kita tidak load dataset 1024 baris di sini, kita simulasi visual scatter di sekitar centroid)
        
        # Plot Centroids
        fig = go.Figure()
        
        # Tambahkan Centroid ke Grafik
        for idx, row in df_centers.iterrows():
            cluster_id = row['Cluster_Label']
            fig.add_trace(go.Scatter(
                x=[row['Curah_Hujan']], 
                y=[row['Elevasi']],
                mode='markers',
                marker=dict(size=25, color=color_map[cluster_id], opacity=0.3),
                name=f"Pusat {risk_map[cluster_id]}"
            ))

        # Tambahkan Input User
        fig.add_trace(go.Scatter(
            x=[val_hujan],
            y=[val_elevasi],
            mode='markers+text',
            marker=dict(size=15, color='blue', line=dict(width=2, color='DarkSlateGrey')),
            text=["📍 Lokasi Anda"],
            textposition="top center",
            name="Input Data"
        ))

        fig.update_layout(
            title="Peta Sebaran Klaster (Curah Hujan vs Elevasi)",
            xaxis_title="Curah Hujan (mm)",
            yaxis_title="Elevasi (mdpl)",
            template="plotly_white",
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"**Analisa:** Daerah dengan elevasi **{val_elevasi} mdpl** dan curah hujan **{val_hujan} mm** dikategorikan sebagai **{result_text}**. Disarankan untuk melakukan pengecekan drainase rutin di area {input_kecamatan}.")

with tab2:
    st.header("Detail Pusat Klaster (Centroids)")
    st.markdown("Tabel berikut menunjukkan karakteristik rata-rata dari setiap kategori risiko yang dipelajari oleh Machine Learning.")
    
    # Format tabel untuk tampilan
    df_display = df_centers.copy()
    df_display['Kategori Risiko'] = df_display['Cluster_Label'].map(risk_map)
    df_display = df_display[['Kategori Risiko', 'Curah_Hujan', 'Elevasi']]
    df_display.columns = ['Kategori Risiko', 'Rata-rata Curah Hujan (mm)', 'Rata-rata Elevasi (mdpl)']
    
    st.table(df_display)
    
    st.markdown("""
    **Penjelasan:**
    * **Sangat Rawan:** Biasanya terjadi di daerah pesisir atau cekungan rendah (Banda Sakti, sebagian Blang Mangat) dengan curah hujan tinggi.
    * **Waspada:** Daerah transisi yang masih berpotensi banjir genangan.
    * **Aman:** Daerah perbukitan atau dataran tinggi (sebagian Muara Satu/Dua).

    """)


