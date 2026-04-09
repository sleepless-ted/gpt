import re
from nltk.stem import WordNetLemmatizer

pattern = pattern = r"[a-z]+'[a-z]+|[a-z0-9]+(?:\.[0-9]+)?|[.,!?;:]"
_lemmatizer = WordNetLemmatizer()

def tokenize(text, pattern):
    text = text.lower()
    tokens = re.findall(pattern, text)
    return [_lemmatizer.lemmatize(token) for token in tokens]