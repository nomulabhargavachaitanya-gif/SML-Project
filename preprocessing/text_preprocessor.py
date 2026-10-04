import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

try:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
except Exception as e:
    print(f"NLTK Download Warning: {e}")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        try:
            text = text.lower()
            # Soft regex to keep numbers, crucial for Tech/Business categories
            text = re.sub(r'[^a-z0-9\s]', '', text) 
            tokens = word_tokenize(text)
            filtered = [w for w in tokens if w not in self.stop_words]
            return " ".join(filtered)
        except Exception as e:
            print(f"Error cleaning text: {e}")
            return text

    def transform_series(self, series):
        return series.apply(self.clean_text)