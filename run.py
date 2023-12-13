from llm import LlmInference
from retrieval.retrieval import Retrieval
#import googletrans
import streamlit as st

class Inference:
    def __init__(
            self,
            url,
            header,
            data_path='raw_data/*/*.txt',
            output_path='retrieval/saved',
            reset=False
    ):
        self.llm = LlmInference(url=url, header=header)
        self.doc_retrieval = Retrieval(data_path=data_path, output_path=output_path,reset=reset)
        #self.translator = googletrans.Translator()

    def __call__(self, question):
        informations, _, _, _ = self.doc_retrieval(question=question)
        if informations != []:
            joint_information = '\n'.join(informations)
            #joint_information = joint_information.replace('||', '\n')
            #answer = self.llm(question=question, contexts=joint_information)
            answer = ''
        else:
            answer = "Không có tài liệu liên quan được tìm thấy trong bộ dữ liệu hiện tại !"
            informations = ['', '', '']
        #answer = self.translator.translate(answer ,src='en' ,dest='vi').text
        return answer, informations