import re
from typing import List, Iterable, Optional
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- NLTK stopwords handling (download bila perlu) ---
try:
    from nltk.corpus import stopwords
except Exception:
    import nltk as _nltk
    _nltk.download('stopwords')
    from nltk.corpus import stopwords

try:
    STOPWORDS_ID = set(stopwords.words('indonesian'))
except LookupError:
    import nltk as _nltk
    _nltk.download('stopwords')
    STOPWORDS_ID = set(stopwords.words('indonesian'))

# --- Stemmer Sastrawi ---
_stemmer = StemmerFactory().create_stemmer()

# --- Contoh kamus slang sederhana (tambahkan sesuai kebutuhan) ---
# Untuk proyek nyata, simpan kamus slang di file terpisah dan load di sini.
SLANG_DICT = {
    'gk': 'tidak',
    'ga': 'tidak',
    'gak': 'tidak',
    'gatau': 'tidak tahu',
    'tdk': 'tidak',
    'klo': 'kalau',
    'kpn': 'kapan',
    'sb': 'sebagai',
    'dg': 'dengan',
    'dgn': 'dengan',
    'yg': 'yang',
    'td': 'tidak',
    'mksh': 'terima kasih',
    'thx': 'terima kasih',
    'btw': 'omong-omong'
}

# --- Regex patterns ---
URL_PATTERN = re.compile(r'http\S+|www\.\S+')
HTML_TAG_PATTERN = re.compile(r'<.*?>')
NON_ALPHABETIC_PATTERN = re.compile(r'[^a-z\s]')  # hanya huruf a-z dan spasi
MULTISPACE_PATTERN = re.compile(r'\s+')
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE
)

# -- ganti definisi EMOTICON_MAP dan map_emoticons dengan kode ini --

# peta emoticon -> kata (gunakan emoticon literal, tanpa backslash escape)
EMOTICON_MAP = {
    ":)": "senang",
    ":-)": "senang",
    ":D": "senang",
    ":(": "sedih",
    ":-(": "sedih",
    ";)": "senang",
    ":'(": "sedih"
}

# Jika sebelumnya ada EMOTICON_RE, hapus atau komentar baris pembuatannya.
# Kita tidak lagi memakai regex untuk emoticon supaya lebih aman.

def map_emoticons(text: str) -> str:
    """
    Ganti emoticon ASCII menjadi kata. Tidak memakai regex — aman terhadap
    karakter khusus.
    """
    if not text:
        return text
    for emot, rep in EMOTICON_MAP.items():
        # tambahkan spasi di sekitar pengganti untuk menghindari penggabungan kata
        text = text.replace(emot, " " + rep + " ")
    return text




# -------------------------
# Helper functions
# -------------------------
def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(' ', text)


def remove_html(text: str) -> str:
    return HTML_TAG_PATTERN.sub(' ', text)


def remove_emoji(text: str) -> str:
    # hapus rentang unicode emoji
    text = EMOJI_PATTERN.sub(' ', text)
    # hapus non-ascii leftover
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text

def normalize_slang(text: str, slang_dict: Optional[dict] = None) -> str:
    if not slang_dict:
        slang_dict = SLANG_DICT
    tokens = text.split()
    normalized = [slang_dict.get(t, t) for t in tokens]
    return ' '.join(normalized)


def remove_non_alphabetic(text: str, keep_spaces: bool = True) -> str:
    # buang angka dan tanda baca; sisakan huruf a-z (english lowercase)
    return NON_ALPHABETIC_PATTERN.sub(' ', text) if keep_spaces else NON_ALPHABETIC_PATTERN.sub('', text)


def tokenize(text: str) -> List[str]:
    return text.split()


def remove_stopwords(tokens: Iterable[str], extra_stopwords: Optional[Iterable[str]] = None) -> List[str]:
    extra = set(extra_stopwords) if extra_stopwords else set()
    return [t for t in tokens if t and (t not in STOPWORDS_ID) and (t not in extra)]


def stem_tokens(tokens: Iterable[str]) -> List[str]:
    # Sastrawi expects a string; stem per token for clarity
    return [_stemmer.stem(t) for t in tokens]


# -------------------------
# Main pipeline
# -------------------------
def preprocess_text(
    text: str,
    lower: bool = True,
    remove_urls_flag: bool = True,
    remove_html_flag: bool = True,
    map_emoticons_flag: bool = True,
    remove_emoji_flag: bool = True,
    normalize_slang_flag: bool = True,
    remove_non_alpha_flag: bool = True,
    remove_stopwords_flag: bool = True,
    extra_stopwords: Optional[Iterable[str]] = None,
    do_stemming: bool = True
) -> str:

    if not isinstance(text, str):
        return ''

    # 1. lowercase
    if lower:
        text = text.lower()

    # 2. basic cleaning
    if remove_urls_flag:
        text = remove_urls(text)
    if remove_html_flag:
        text = remove_html(text)
    if map_emoticons_flag:
        text = map_emoticons(text)
    if remove_emoji_flag:
        text = remove_emoji(text)

    # 3. normalisasi slang (sebelum menghapus non-alpha agar 'gk' => 'tidak')
    if normalize_slang_flag:
        text = normalize_slang(text, SLANG_DICT)

    # 4. buang karakter non-alfabet
    if remove_non_alpha_flag:
        text = remove_non_alphabetic(text)

    # 5. normalisasi spasi
    text = MULTISPACE_PATTERN.sub(' ', text).strip()

    if not text:
        return ''

    # 6. tokenize
    tokens = tokenize(text)

    # 7. stopword removal
    if remove_stopwords_flag:
        tokens = remove_stopwords(tokens, extra_stopwords)

    # 8. stemming
    if do_stemming:
        tokens = stem_tokens(tokens)

    return ' '.join(tokens)


def preprocess_corpus(
    texts: Iterable[str],
    **kwargs
) -> List[str]:
    return [preprocess_text(t, **kwargs) for t in texts]

if __name__ == '__main__':
    examples = [
        "Saya suka ChatGPT 😊. Fitur-nya mantap! http://example.com",
        "Gk ngerti pakai ini, error terus :( ",
        "<b>Aplikasi</b> bagus tapi ada iklan",
        "Kapan update nya? btw thx"
    ]

    print("Contoh preprocessing:")
    for ex in examples:
        cleaned = preprocess_text(ex)
        print("ORIG :", ex)
        print("CLEAN:", cleaned)
        print("---")