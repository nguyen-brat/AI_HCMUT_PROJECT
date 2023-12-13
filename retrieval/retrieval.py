from .tfidf import SparseRetrieval
from .embedding import EmbeddingRetrieval
from sentence_transformers import CrossEncoder
import torch
import numpy as np
from glob import glob
import re
from underthesea import sent_tokenize, word_tokenize

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def frobenius_normalize(z):
    normalized_z = z / np.linalg.norm(z)
    return normalized_z

class Retrieval:
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
                self.data_content.append(f.read())
        self.dense_retrieval = EmbeddingRetrieval(data_path=data_path, output_path=output_path, reset=reset)
        self.sparse_retrieval = SparseRetrieval(data_path=data_path, output_path=output_path, reset=reset)
        self.reranking_model = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', num_labels=2, max_length=256, device='cpu')
        #self.reranking_model = CrossEncoder('nguyen-brat/pdt-reranking-model_v2', num_labels=2, max_length=256, device='cpu')

    def __call__(self, question, k=3, alpha=0.5, tokenize=False, rerank=True):
        top_result, sparse_score, dense_score, total_score = self.retrieval_(question, k, alpha, tokenize)
        documents = [self.data_content[idx] for idx, score in zip(top_result, total_score) if score > 0.2]
        if rerank:
            documents, sort_index = self.reranking_inference(question=question, fact_list=documents, tokenize=False)
        return documents, sparse_score[sort_index].tolist(), dense_score[sort_index].tolist(), total_score[sort_index].tolist()


    def retrieval_(self, question, k=3, alpha = 0.5, tokenize=False):
        _, sparse_score = self.sparse_retrieval.retrieval_(query=question, k=k)
        if tokenize:
            question = word_tokenize(question, format='text')
        _, dense_score = self.dense_retrieval.retrieval_(question=question, k=k)
        scores = dense_score*(1-alpha) + sparse_score*alpha
        sort_index = np.argsort(scores)[::-1][:k]
        return sort_index, sparse_score[sort_index], dense_score[sort_index], scores[sort_index]
    
    def reranking_inference(self, question:str, fact_list, tokenize=False, max_length=256):
        '''
        take claim and list of fact list
        return reranking fact list and score of them
        '''
        scores = []
        question = self.clean(question)
        for fact in fact_list:
            scores.append(self.reranking_inference_(question, self.clean(fact), max_length, tokenize))
        sort_index = np.argsort(np.array(scores))
        reranking_answer = list(np.array(fact_list)[sort_index])
        reranking_answer.reverse()
        return reranking_answer, sort_index
    
    def reranking_inference_(self, question:str, fact:str, max_length=256, tokenize=False):
        facts = self.split_doc(fact, max_length, tokenize)
        scores = []
        for afact in facts:
            with torch.no_grad():
                result = softmax(self.reranking_model.predict([question, afact]))[1]
                scores.append(result)
        return max(scores)
    
    @staticmethod
    def split_doc(document, max_length=256, tokenize=False):
        if tokenize:
            document = word_tokenize(document, format='text')
        #print(document)
        document_sentences = sent_tokenize(document)
        result = []
        sentences = []
        total_len = 0
        for sentence in document_sentences:
            total_len += len(sentence.split())
            if total_len < max_length:
                sentences.append(sentence)
            elif len(sentence.split()) < max_length: # if that sentence is not longer than  max length
                result.append(' '.join(sentences))
                sentences = [sentence]
                total_len = len(sentence.split())
            else: # if a sentence is longer than max length just cut that sentence off
                result.append(' '.join(sentence.split()[:max_length]))
                sentences = []
                total_len = 0
        if sentence != []:
            result.append(' '.join(sentences))
        return result
    
    @staticmethod
    def clean(text):
        for match in re.finditer(r"(\d\.\d|)(\w\.\w)", text):
            text = text[:match.span()[0]+1] + '|' + text[match.span()[1]-1:]
        text = re.sub(r'\n+', r'.', text)
        text = re.sub(r';', r'.', text)
        text = re.sub(r'\.+', r'. ', text)
        #text = re.sub(r"['\",\?:\-!-]", "", text)
        text = text.replace('|', '.')
        text = text.strip()
        text = " ".join(text.split())#.lower()
        return text