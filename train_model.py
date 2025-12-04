import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from preprocessing import preprocess_text

DATA_CSV = 'data/chatgpt_reviews_labeled.csv'
MODEL_OUT = 'models/sentiment_model.pkl'
VECT_OUT = 'models/tfidf.pkl'


def load_and_prepare():
    df = pd.read_csv(DATA_CSV)

    if 'sentiment' not in df.columns:
        # heuristik sederhana berdasarkan skor (jika tersedia)
        if 'score' in df.columns:
            df['sentiment'] = df['score'].apply(lambda x: 'positif' if x>=4 else ('negatif' if x<=2 else 'netral'))
        else:
            raise ValueError("Data belum memiliki label sentiment. Tambahkan kolom 'sentiment'.")

    df['clean'] = df['content'].astype(str).apply(preprocess_text)
    # drop baris kosong
    df = df[df['clean'].str.strip()!='']
    return df


def train():
    df = load_and_prepare()
    X = df['clean']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LogisticRegression(max_iter=400, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds))

    # simpan model pipeline
    joblib.dump(pipeline, MODEL_OUT)
    print(f"Model tersimpan di {MODEL_OUT}")


if __name__ == '__main__':
    train()