import streamlit as st
import importlib
import json
import os

st.set_page_config(page_title="Analisis Sentimen — ChatGPT Reviews", layout="centered")

# ---------- coba import backend bert, jika gagal fallback ke tf-idf ----------
backend = None
predict_fn = None  # fungsi yang akan dipanggil: predict(text) -> (label, {class:prob})

# coba IndoBERT
try:
    pb = importlib.import_module("predict_bert")
    predict_fn = getattr(pb, "predict_text_bert")
    backend = "IndoBERT"
except Exception:
    # fallback ke TF-IDF pipeline
    try:
        pp = importlib.import_module("predict")
        # predict.predict_text(text) -> (label, probs_dict) di versi TF-IDF
        predict_fn = getattr(pp, "predict_text")
        backend = "TF-IDF pipeline"
    except Exception:
        backend = None
        predict_fn = None

# header
st.title("Analisis Sentimen — ChatGPT Reviews")
if backend:
    st.caption(f"Model backend aktif: {backend}")
else:
    st.error(
        "Tidak ada backend prediksi ditemukan. "
        "Pastikan salah satu dari berikut tersedia:\n"
        "- models/indobert-sentiment (hasil train IndoBERT) dan file predict_bert.py, OR\n"
        "- models/sentiment_model.pkl dan file predict.py (TF-IDF pipeline)."
    )
    st.stop()

# input
text = st.text_area("Masukkan ulasan (Indonesia)", height=150)

col1, col2 = st.columns([1, 4])
with col1:
    predict_btn = st.button("Prediksi")
with col2:
    st.write("")  # ruang

# helper untuk menampilkan probabilitas rapi
def pretty_probs(probs_dict):
    # urutkan descending
    items = sorted(probs_dict.items(), key=lambda x: -x[1])
    # kembalikan mapping yang rapi (label: round)
    return {k: round(v, 6) for k, v in items}

if predict_btn:
    if not text or not text.strip():
        st.warning("Masukkan teks untuk dites.")
    else:
        try:
            label, probs = predict_fn(text)
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")
        else:
            st.markdown("### Hasil:")
            st.write(f"**{label}**")
            st.markdown("**Probabilitas per kelas:**")
            st.json(pretty_probs(probs))
            # tambahan: tampilkan penjelasan singkat kalau menggunakan TF-IDF
            if backend == "TF-IDF pipeline":
                st.info(
                    "Catatan: TF-IDF tidak selalu menangkap negasi/context. "
                    "Jika hasil kurang memuaskan, gunakan IndoBERT (lebih memahami konteks)."
                )

# footer / petunjuk
# st.markdown("---")
# st.markdown(
#     "Petunjuk:\n\n"
#     "- Jika ingin memakai IndoBERT, pastikan folder `models/indobert-sentiment` berisi model & tokenizer (hasil `train_bert.py`) dan file `predict_bert.py` ada di project.\n"
#     "- Jika ingin memakai pipeline TF-IDF, pastikan file `models/sentiment_model.pkl` & `models/tfidf.pkl` tersedia dan file `predict.py` ada di project.\n"
#     "- Jalankan aplikasi dengan: `streamlit run app.py`"
# )
