import pandas as pd

df = pd.read_csv('data/chatgpt_reviews.csv')

def map_score_to_sentiment(s):
    try:
        s = float(s)
    except:
        return 'netral'
    if s >= 4:
        return 'positif'
    if s <= 2:
        return 'negatif'
    return 'netral'

df['sentiment'] = df['score'].apply(map_score_to_sentiment)

df.to_csv('data/chatgpt_reviews_labeled.csv', index=False)
print("Selesai membuat label sentiment!")
print(df['sentiment'].value_counts())
