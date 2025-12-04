import joblib
from preprocessing import preprocess_text

MODEL_OUT = 'models/sentiment_model.pkl'


def predict_text(text: str):
    pipeline = joblib.load(MODEL_OUT)
    clean = preprocess_text(text)
    pred = pipeline.predict([clean])[0]
    probs = pipeline.predict_proba([clean])[0]
    classes = pipeline.classes_
    prob_dict = dict(zip(classes, probs))
    return pred, prob_dict


if __name__ == '__main__':
    contoh = "Aplikasi ini sangat membantu, respons cepat dan akurat"
    label, probs = predict_text(contoh)
    print('Teks:', contoh)
    print('Prediksi:', label)
    print('Probabilitas:', probs)