from sklearn.feature_extraction.text import TfidfVectorizer
from glob import glob
import numpy as np
import pickle
import os
import re
from sklearn.metrics.pairwise import cosine_similarity

class SparseRetrieval:
    def __init__(
            self,
            data_path='raw_data/*/*.txt',
            output_path='retrieval/saved',
            reset=False
    ):
        self.data_paths = glob(data_path)
        self.data_paths.sort()
        self.data_content = []
        for data in self.data_paths:
            with open(data, 'r') as f:
                self.data_content.append(self.clean(f.read()))
        
        if reset:
            all_file_paths = glob(output_path + '/*')
            for file_name in all_file_paths:
                os.remove(file_name)
            os.remove(output_path) 

        if not os.path.exists(output_path+'/tfidf_vectorizer.pkl'):
            self.vectorizer = TfidfVectorizer(input='content', ngram_range = (1, 3), token_pattern=r"(?u)\b[\w\d]+\b")
            self.corpus_vectorize = self.vectorizer.fit_transform(self.data_content)
            self.save(output_path=output_path)
        else:
            with open(os.path.join(output_path, 'tfidf_vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(output_path, 'tfidf_corpus_vector.pkl'), 'rb') as f:
                self.corpus_vectorize = pickle.load(f)

    def retrieval_(self, query, k=3):
        query_vector = self.vectorizer.transform([self.clean(query)])
        similar = cosine_similarity(query_vector, self.corpus_vectorize)[0]
        sort_index = np.argsort(similar)[::-1][:k]
        return sort_index, similar
    
    def __call__(self, query, k=3):
        top_index, _ = self.retrieval_(query=self.clean(query), k=5)
        result = []
        for index in top_index:
            with open(self.data_paths[index], 'r') as f:
                result.append(f.read())
        return result[:k]
    
    @staticmethod
    def clean(text):
        for match in re.finditer(r"(\d\.\d|)(\w\.\w)", text):
            text = text[:match.span()[0]+1] + '|' + text[match.span()[1]-1:]
        text = re.sub(r'\n+', r'.', text)
        text = re.sub(r';', r'.', text)
        text = re.sub(r'\.+', r' . ', text)
        #text = re.sub(r"['\",\?:\-!-]", "", text)
        text = text.replace('|', '.')
        text = text.strip()
        text = " ".join(text.split()).lower()
        return text
    
    def save(self, output_path='retrieval/saved'):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        with open(os.path.join(output_path, 'tfidf_vectorizer.pkl'), "wb") as file:
            pickle.dump(self.vectorizer, file)
        with open(os.path.join(output_path, 'tfidf_corpus_vector.pkl'), "wb") as file:
            pickle.dump(self.corpus_vectorize, file)