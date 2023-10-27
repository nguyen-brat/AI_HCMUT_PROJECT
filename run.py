from llm import LlmInference
from retrieval.tfidf import DocIR
import streamlit as st

class Inference:
    def __init__(
            self,
            url = "https://bahnar.dscilab.com:20007/llama/api",
            header = {"Content-Type": "application/json"},
            data_path='raw_data/*/*.txt',
            output_path='retrieval/saved',
            reset=False
    ):
        self.llm = LlmInference(url=url, header=header)
        self.doc_retrieval = DocIR(data_path=data_path, output_path=output_path,reset=reset)

    def __call__(self, question):
        informations = self.doc_retrieval(query=question)
        print(informations)
        joint_information = '\n'.join(informations)
        answer = self.llm(question=question, contexts=joint_information)
        return answer, informations