import re
pattern = r"[a-z0-9&]+"

def tokenize(text, pattern):
    text = text.lower()
    return re.findall(pattern, text)