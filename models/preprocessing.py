import re
import nltk


def lemmanizer(chunk):
    lemma = nltk.WordNetLemmatizer()
    return " ".join([lemma.lemmatize(word) for word in chunk])


def preprocess(df):
    # слова, которые можно умышленно отнести к стопворд не удалены сейм с пунктуацией
    return df['excerpt'].apply(lambda x: re.sub(r'\s\s+', ' ', lemmanizer(nltk.word_tokenize(x.lower()))))
