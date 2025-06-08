import re
from typing import List
from collections import defaultdict
import multiprocessing
from tqdm import tqdm
import spacy
from pyvi import ViTokenizer

LANG_CLS = defaultdict(lambda:"VietnameseTokenizer")
LANG_CLS.update({
    "en": "SpacyTokenizer",
    "vi": "VietnameseTokenizer",
})

SPACY_MODEL = {
    "en": "en_core_web_sm",
    "ja": "ja_core_news_sm"
}

class SpacyTokenizer(object):
    def __init__(self, lang="en", stopwords=None):
        self.stopwords = stopwords
        self.nlp = spacy.load(SPACY_MODEL[lang], disable=['ner', 'parser'])
        print("Using SpaCy tokenizer")

        
    def tokenize(self, lines: List[str]) -> List[List[str]]:
        docs = self.nlp.pipe(lines, batch_size=1000, n_process=multiprocessing.cpu_count())
        docs = [[token.lemma_ for token in doc if not (token.is_stop or token.is_punct)] for doc in docs]
        return docs
class VietnameseTokenizer(object):
    def __init__(self, stopwords=None):
        self.pat = re.compile(r'[0-9!"#$%&\'()*+,-./:;<=>?@—，。：★、￥…【】（）《》？“”‘’！\[\\\]^`{|}~\u3000]+')
        self.stopwords = stopwords
        print("Using Vietnamese tokenizer (pyvi)")
    
    def tokenize(self, lines: List[str]) -> List[List[str]]:
        docs = []
        for line in tqdm(lines):
            # Tách từ bằng pyvi, giữ dấu _
            tokenized_text = ViTokenizer.tokenize(line.lower())
            # Tách token nhưng giữ từ ghép chứa _
            tokens = re.split(r'\s+', tokenized_text.strip())
            # Loại bỏ ký tự đặc biệt nhưng không ảnh hưởng đến _
            tokens = [re.sub(self.pat, '', t).strip() for t in tokens if t]
            # Loại bỏ token rỗng hoặc quá ngắn
            tokens = [t for t in tokens if len(t) > 1]
            # Loại bỏ stopwords
            if self.stopwords is not None:
                tokens = [t for t in tokens if t not in self.stopwords]
            docs.append(tokens)
        return docs

if __name__ == '__main__':
    # Test VietnameseTokenizer with file input
    vi_tokenizer = VietnameseTokenizer()
    file_path = '/content/Neural_Topic_Models/data/processed_articles.txt'
    # try:
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         lines = [line.strip() for line in f if line.strip()]  # Read and strip empty lines
    #     print("Vietnamese result:", vi_tokenizer.tokenize(lines[:5]))  # Tokenize first 5 lines for brevity
    # except FileNotFoundError:
    #     print(f"Error: File not found at {file_path}")
    # except Exception as e:
    #     print(f"Error: {str(e)}")