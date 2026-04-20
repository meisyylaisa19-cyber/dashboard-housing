import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score

# ================= PAGE CONFIG (HARUS SEKALI & DI ATAS) =================
st.set_page_config(page_title="Dashboard Housing", layout="wide")

# STYLE
sns.set_style("whitegrid")

st.title("🔥 STREAMLIT BERJALAN")
st.success("Kalau ini muncul, berarti sistem OK")
st.write("Deploy berhasil 🎉")

# ================= SIDEBAR =================
st.sidebar.title("📊 Dashboard Menu")
menu = st.sidebar.radio("Menu", ["EDA", "Clustering", "Classification", "Kesimpulan"])
file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

# ================= JIKA BELUM UPLOAD =================
if file is None:
    st.warning("Silakan upload dataset terlebih dahulu")
    st.stop()

# ================= LOAD DATA =================
df = pd.read_csv(file)

# ================= FILTER =================
st.sidebar.markdown("## 🎛️ Filter Data")

min_income, max_income = st.sidebar.slider(
    "Median Income",
    float(df['median_income'].min()),
    float(df['median_income'].max()),
    (float(df['median_income'].min()), float(df['median_income'].max()))
)

min_age, max_age = st.sidebar.slider(
    "Housing Median Age",
    int(df['housing_median_age'].min()),
    int(df['housing_median_age'].max()),
    (int(df['housing_median_age'].min()), int(df['housing_median_age'].max()))
)

if 'ocean_proximity' in df.columns:
    lokasi = st.sidebar.multiselect(
        "Ocean Proximity",
        df['ocean_proximity'].unique(),
        default=df['ocean_proximity'].unique()
    )

    df = df[
        (df['median_income'].between(min_income, max_income)) &
        (df['housing_median_age'].between(min_age, max_age)) &
        (df['ocean_proximity'].isin(lokasi))
    ]

# ================= HEADER =================
st.title("🏘️California Housing Dashboard🏘️")
st.markdown("---")

# ================= EDA =================
if menu == "EDA":

    st.subheader("📈Exploratory Data Analysis📉")
    st.dataframe(df.head())

    col1, col2, col3 = st.columns(3)
    col1.metric("Jumlah Data", len(df))
    col2.metric("Jumlah Fitur", df.shape[1])
    col3.metric("Missing Value", df.isnull().sum().sum())

# ================= CLUSTERING =================
elif menu == "Clustering":

    st.subheader("🔵 K-Means Clustering")

    fitur = st.multiselect(
        "Pilih fitur",
        df.select_dtypes(include=['int64', 'float64']).columns
    )

    if len(fitur) > 0:

        X = df[fitur]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KMeans(n_clusters=4, random_state=42)
        df['Cluster'] = model.fit_predict(X_scaled)

        sil_score = silhouette_score(X_scaled, df['Cluster'])
        st.metric("Silhouette Score", f"{sil_score:.3f}")

        if len(fitur) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(x=X[fitur[0]], y=X[fitur[1]], hue=df['Cluster'])
            st.pyplot(fig)

        cluster_counts = df['Cluster'].value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8,5))
        ax.bar(cluster_counts.index.astype(str), cluster_counts.values)
        st.pyplot(fig)

        # ================= INSIGHT CLUSTER (TIDAK DIRUBAH) =================
        st.markdown("### 🧠 Insight Cluster")
        st.info("""
Hasil clustering menunjukkan adanya empat segmentasi (4 cluster) utama berdasarkan karakteristik wilayah perumahan.Cluster terbentuk berdasarkan kombinasi faktor lokasi, kepadatan, dan tingkat pendapatan. Perbedaan karakteristik keempat cluster tersebut, sebagai berikut:

•	Cluster 0 (sekitar Los Angeles): Cluster ini mempresentasikan pemukiman tua dengan pendapatan menengah ke bawah di wilayah selatan California.

•	Cluster 1 (sekitar Sacramento): Cluster ini merepresentasi pedalaman utara dengan properti tertua dan populasi paling jarang.

•	Cluster 2 (sekitar San Francisco Bay Area): Cluster ini merepresentasikan wilayah perkotaan muda dengan kepadatan tinggi di wilayah utara California.

Namun, nilai Silhouette Score sebesar 0,228 menunjukkan bahwa kualitas pemisahan cluster masih tergolong rendah, sehingga masih terdapat tumpang tindih antar kelompok.
        """)

# ================= CLASSIFICATION =================
elif menu == "Classification":

    st.subheader("🟢 Random Forest Classification")

    df['Category'] = pd.qcut(df['median_house_value'], 3, labels=['Low', 'Medium', 'High'])

    fitur = st.multiselect(
        "Pilih fitur",
        df.columns.drop(['Category', 'median_house_value'])
    )

    if len(fitur) > 0:

        X = df[fitur]
        y = df['Category']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.3f}")

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
        st.pyplot(fig)

        # ================= INSIGHT CLASSIFICATION (TIDAK DIRUBAH) =================
        st.markdown("### 🧠 Insight Classification")
        st.success("""
Model Random Forest menunjukkan performa yang baik dengan akurasi sekitar 80%, hal ini berarti data berhasil diklasifikasikan dengan benar. Hal ini didukung oleh visualisasi dengan Feature Importance yaitu variabel median_income menjadi faktor paling berpengaruh dalam menentukan harga rumah, diikuti oleh variabel lokasi(latitude dan longtitude). 

Evaluasi model menggunakan confusion matrix yang menunjukkan bahwa masih terdapat kesalahan klasifikasi, terutama antara kelas High ke Medium serta Low ke High. Meskipun demikian, model sudah cukup mampu menangkap pola dalam data, walaupun masih terdapat tumpang tindih antar kelas yang menyebabkan beberapa kesalahan prediksi yang kemungkinan diakibatkan adanya overlap karakteristik.
        """)

# ================= KESIMPULAN =================
elif menu == "Kesimpulan":

    st.subheader("🧠 Kesimpulan")

    st.write("""
Hasil evaluasi model menunjukkan adanya perbedaan kinerja antara metode clustering dan classification. Pada metode K-Means Clustering, diperoleh nilai Silhouette Score sebesar 0,228 yang tergolong rendah. Menunjukkan struktur cluster yang terbentuk belum terpisah secara optimal. Hal ini mengindikasikan struktur antar cluster tidak terpisah dengan baik dan masih banyak tumpang tindih antar segmen. Rendahnya skor model, dapat disebabkan oleh keterbatasan fitur yang digunakan (hanya 6 variabel) serta kompleksitas alami data. Meskipun pada visualisasi terlihat adanya pola geografis, khususnya lokasi, namun secara statistik kualitas pengelompokan masih rendah. 

Sebaliknya, model Random Forest untuk klasifikasi harga rumah menunjukkan performa yang jauh lebih baik dengan akurasi 80,81%. Hasil evaluasi lebih lanjut melalui classification report dan confusion matrix menunjukkan bahwa model cukup stabil dalam mengklasifikasikan setiap kategori, meskipun masih terdapat kesalahan pada kelas dengan karakteristik yang mirip. Selain itu, analisis feature importance menunjukkan bahwa variabel median_income serta lokasi geografis (longitude dan latitude) merupakan faktor yang paling dominan dalam menentukan kategori harga rumah. 

Jika dibandingkan antara kedua metode pada dataset ini, metode classification lebih unggul karena mampu memanfaatkan seluruh variabel serta menangkap hubungan non-linear dalam data, sedangkan clustering menunjukkan performa yang lemah. Oleh karena itu, perbedaan jumlah variabel dan karakteristik metode menjadi faktor utama yang mempengaruhi kualitas hasil yang diperoleh.

Berdasarkan hasil analisis tersebut, dapat disimpulkan bahwa model klasifikasi Random Forest (akurasi 80,81%) jauh lebih unggul dibandingkan model clustering K-Means (Silhouette Score 0,228) dalam memproses data perumahan ini. 
    """)

    st.success("Kesimpulan: Random Forest lebih unggul dibanding K-Means.")
