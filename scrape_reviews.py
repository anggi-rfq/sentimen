from google_play_scraper import reviews, Sort
import pandas as pd
import time

APP_ID = 'com.openai.chatgpt'  # ganti jika perlu
OUT_CSV = 'data/chatgpt_reviews.csv'


def scrape_reviews(count=5000, lang='id', country='id'):
    all_reviews = []
    batch = 200  # google_play_scraper cepat mengambil batch
    fetched = 0

    while fetched < count:
        take = min(batch, count - fetched)
        result, _ = reviews(
            APP_ID,
            lang=lang,
            country=country,
            sort=Sort.NEWEST,
            count=take,
            continuation_token=None
        )
        if not result:
            break
        all_reviews.extend(result)
        fetched += len(result)
        time.sleep(1)

    df = pd.DataFrame(all_reviews)
    # simpan kolom penting
    df = df[['userName', 'content', 'score', 'at', 'replyContent', 'reviewId']]
    df.to_csv(OUT_CSV, index=False)
    print(f"Selesai: {len(df)} review tersimpan di {OUT_CSV}")


if __name__ == '__main__':
    scrape_reviews(count=3000)
