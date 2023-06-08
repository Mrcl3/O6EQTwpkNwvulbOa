
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from unidecode import unidecode
from gensim.models import Word2Vec
from nltk.stem import WordNetLemmatizer
import string
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DataProcessor():
    def __init__(self,df) -> None:
            self.df = df
            

    def preprocess(self, sentence):
        # Converting the words to lowercase]
        sentence = sentence.lower()
        #Drop some columns

        # Collecting a list of stop words from nltk and punctuation from the string class
        stopset = stopwords.words('english') + list(string.punctuation)
        
        # Remove stop words and punctuations from the string
        words = word_tokenize(sentence)
        words = [i for i in words if i.lower() not in stopset and not any(char in string.punctuation for char in i)]
        new_sentence = " ".join(words)
        
        # Remove non-ASCII characters
        new_sentence = unidecode(new_sentence)
        
        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        words = word_tokenize(new_sentence)
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return " ".join(words)
    
    
    def calculate_similarity(self, input_sentence):
        model = self.train_word2vec_model()
        similarity_scores = self.calculate_similarity_scores(input_sentence, model)
        self.rank_data(similarity_scores)
        return self.df

    def train_word2vec_model(self):
        sentences = self.df['job_title'].apply(nltk.word_tokenize).values.tolist()
        model = Word2Vec(sentences, min_count=1, vector_size=300, workers=4)
        return model

    def calculate_similarity_scores(self, input_sentence, model):
        similarity_array = []
        for item in self.df['job_title'].values:
            tokens1 = nltk.word_tokenize(item)
            tokens2 = nltk.word_tokenize(input_sentence)
            similarity = model.wv.n_similarity(tokens1, tokens2)
            similarity_array.append(similarity)
        return similarity_array
    def rank_data(self, similarity_scores):
        features = ['similarities', 'connection']
        self.df['connection'] = self.df['connection'].replace('500+ ', '500')
        self.df['connection'] = pd.to_numeric(self.df['connection'])
        scaler = MinMaxScaler()
        self.df['connection'] = scaler.fit_transform(self.df['connection'].values.reshape(-1, 1))
        similarity_scores = scaler.fit_transform(np.array(similarity_scores).reshape(-1, 1))
        self.df['similarities'] = similarity_scores
        self.df['ranking'] = 0.8 * self.df['similarities'] * 0.2 * self.df['connection']
        self.df['ranking'] = scaler.fit_transform(self.df['ranking'].values.reshape(-1, 1))
        self.df['ranking'] = self.df['ranking'].apply(lambda x: round(x, 2))  # Show only two digits after the decimal point
        self.df['connection'] = self.df['connection'].apply(lambda x: round(x, 2))  # Round to two decimal places
        self.df['similarities'] = self.df['similarities'].apply(lambda x: round(x, 2))  # Round to two decimal places
        self.df = self.df.sort_values('ranking', ascending=False)



    def calculate_similarity_scores_sbert(self, input_sentence):
        model = SentenceTransformer('paraphrase-distilroberta-base-v1')
        similarity_array = []
        target_embedding = model.encode([input_sentence])[0]
        sentence_embeddings = model.encode(self.df['job_title'].values)
        similarities = cosine_similarity([target_embedding], sentence_embeddings)[0]
        similarity_array.extend(similarities)
        return similarity_array
    
    def calculate_similarity_bert(self, input_sentence):
        similarity_scores = self.calculate_similarity_scores_sbert(input_sentence)
        self.rank_data(similarity_scores)
        return self.df