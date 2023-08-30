import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

st.set_page_config(page_title="Prediksi penyakit jantung", layout="wide")
st.write("""
# Selamat datang pada pengecekan dini penyakit jantung

Dashboard ini dibuat oleh: [Muhammad Alwi Zahidan](https://www.linkedin.com/in/malwizahidan/)
""")

st.write("""
    Aplikasi ini menyediakan prediksi dini mengenai penyakit jantung berdasarkan gejala-gejala yang dialami pengguna.

    Analisa ini didasarkan pada data yang diperoleh dari [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML.
    """)
st.sidebar.header('Masukkan data pengguna:')
# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Unggah file CSV", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        st.sidebar.header('Masukkan data secara manual')
        cp = st.sidebar.slider('Tipe gejala sakit dada', 1,4,2)
        if cp == 1.0:
            wcp = "Nyeri dada tipe angina"
        elif cp == 2.0:
            wcp = "Nyeri dada tipe nyeri tidak stabil"
        elif cp == 3.0:
            wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
        else:
            wcp = "Nyeri dada yang tidak terkait dengan masalah jantung"
        st.sidebar.write("Jenis nyeri dada yang dirasakan oleh pasien", wcp)
        thalach = st.sidebar.slider("Denyut jantung maksimum", 71, 202, 80)
        slope = st.sidebar.slider("Kemiringan segmen ST pada elektrokardiogram (EKG)", 0, 2, 1)
        oldpeak = st.sidebar.slider("Seberapa banyak ST segmen menurun atau depresi", 0.0, 6.2, 1.0)
        exang = st.sidebar.slider("Angina akibat olahraga", 0, 1, 1)
        ca = st.sidebar.slider("Jumlah pembuluh besar", 0, 3, 1)
        thal = st.sidebar.slider("Hasil tes thalium", 1, 3, 1)
        sex = st.sidebar.selectbox("Jenis Kelamin", ('Perempuan', 'Pria'))
        if sex == "Perempuan":
            sex = 0
        else:
            sex = 1
        age = st.sidebar.slider("Usia", 29, 77, 30)
        data = {'cp': cp,
                'thalach': thalach,
                'slope': slope,
                'oldpeak': oldpeak,
                'exang': exang,
                'ca':ca,
                'thal':thal,
                'sex': sex,
                'age':age}
        features = pd.DataFrame(data, index=[0])
        return features

input_df = user_input_features()
img = Image.open("Coronary-heart-disease.jpg")
st.image(img, width=700)
if st.sidebar.button('Prediksikan'):
    df = input_df
    st.write(df)
    with open("best_model_capstone.pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    prediction = loaded_model.predict(df)
    result = ['tidak berpotensi penyakit jantung' if prediction == 0 else 'berpotensi penyakit jantung']
    st.subheader('Prediksi: ')
    output = str(result[0])
    with st.spinner('Mohon tunggu...'):
        time.sleep(3)
        st.success(f"Hasil prediksi menunjukkan {output}")

