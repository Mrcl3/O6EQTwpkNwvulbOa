import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from unidecode import unidecode
import string
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

