import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])

def tokenize(message):
    """Tokenization function to process the text data"""

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%'\
        '[0-9a-fA-F][0-9a-fA-F]))+'

    # Removing any URLs
    detected_urls = re.findall(url_regex, message)

    for url in detected_urls:
        message = message.replace(url, "urlplaceholder")

    # Removing punctuation
    message = re.sub(r"[^a-zA-Z0-9]", " ", message)

    # Tokenizing the message
    tokens = word_tokenize(message)

    # Lemmatize, lowercase, strip and also removing stopwords
    clean_tokens = [WordNetLemmatizer().lemmatize(t).lower().strip()
                    for t in tokens if t not in stopwords.words("english")]

    return(clean_tokens)
