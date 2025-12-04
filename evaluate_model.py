import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_OUT = 'models/sentiment_model.pkl'
DATA_CSV = 'data/chatgpt_reviews_labeled.csv'


def evaluate():
    pipeline = joblib.load(MODEL_OUT)
    df = pd.read_csv(DATA_CSV)
    df['clean'] = df['content'].astype(str).apply(lambda t: t)  # asumsikan sudah diproses di train
    if 'sentiment' not in df.columns:
        raise ValueError('Dataset tidak memiliki label `sentiment` untuk evaluasi.')

    X = df['clean']
    y = df['sentiment']
    preds = pipeline.predict(X)
    print(classification_report(y, preds))

    cm = confusion_matrix(y, preds, labels=pipeline.classes_)
    print('Confusion matrix:')
    print(cm)

    # visualisasi (opsional)
    try:
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    evaluate()