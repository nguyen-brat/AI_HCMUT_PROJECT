from sentence_transformers import SentenceTransformer
from glob import glob
import os
from tqdm import tqdm
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingRetrieval:
    def __init__(
            self,
            data_path='raw_data/*/*.txt',
            output_path='retrieval/saved',
            reset=False,
    ):
        self.data_paths = glob(data_path)
        self.data_paths.sort()
        self.data_content = []
        for data in self.data_paths:
            with open(data, 'r') as f:
                self.data_content.append(self.clean(f.read()))
        self.model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder', device='cpu')
        self.embedding_path = os.path.join(output_path, 'dense_vector.npy')
        if not os.path.exists(self.embedding_path):
            self.save()
        else:
            self.embedding = np.load(self.embedding_path)

    def __call__(self, question, k=3):
        result = []
        top_index, _ = self.retrieval_(question=question, k=k)
        for index in top_index:
            with open(self.data_paths[index], 'r') as f:
                result.append(f.read())
        return result[:k]
    
    def retrieval_(self, question, k=3):
        question_embed = self.model.encode([question])
        similar_scores = cosine_similarity(question_embed, self.embedding)[0]
        sort_index = np.argsort(similar_scores)[::-1][:k]
        return sort_index, similar_scores
    
    @staticmethod
    def clean(text):
        for match in re.finditer(r"(\d\.\d|)(\w\.\w)", text):
            text = text[:match.span()[0]+1] + '|' + text[match.span()[1]-1:]
        text = re.sub(r'\n+', r'.', text)
        text = re.sub(r';', r'.', text)
        text = re.sub(r'\.+', r'. ', text) #####
        #text = re.sub(r"['\",\?:\-!-]", "", text)
        text = text.replace('|', '.')
        text = text.strip()
        text = " ".join(text.split())
        return text
    
    def save(self):
        embedding = []
        print('Run embedding !')
        for i in tqdm(range(0, len(self.data_content), 10)):
            data = self.data_content[i:i+10]
            embed = self.model.encode(data)
            embedding.append(embed)
        self.embedding = np.concatenate(embedding, axis=0)
        np.save(self.embedding_path, self.embedding)
