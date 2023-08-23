from copy import deepcopy
from re import sub, IGNORECASE

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess(resume_dataframe):
    lemmatizer = WordNetLemmatizer()
    preprocessed_resumes = list()
    for resume in resume_dataframe:
        res = sub(r'\W', ' ', resume)                       # Removing Special Characters
        res = sub(r'(^| ).(( ).)*( |$)', ' ', res)          # Removing single alphabets or characters
        res = sub(r'\s+', ' ', resume, flags=IGNORECASE)    # Removing extra spaces around words
        res = res.split()                                   # Splitting the string into words
        res = [lemmatizer.lemmatize(word) for word in res]  # Lemmatizing the words
        res = ' '.join(res)                                 # Converting the list back to string

        preprocessed_resumes.append(res)                    # appending to list
    return deepcopy(preprocessed_resumes)                   # returning a copy of the list

def get_vectorizer():
    # Returning an initialized tfidf vectorizer
    return TfidfVectorizer(min_df=10, max_df=0.7, stop_words=stopwords.words('english'), lowercase=True)