import requests
import json
import streamlit as st

class LlmInference:
    def __init__(
            self,
            url = "https://bahnar.dscilab.com:20007/llama/api",
            header = {"Content-Type": "application/json"},
    ):
        self.url = url
        self.header = header

    def __call__(self, question, contexts):
        contexts = ' '.join(contexts.split()[:3000])
        info = {
            "prompt": f'''[INST] <<SYS>> Trả lời câu hỏi đưa ra dựa vào văn bản được cung cấp <</SYS>>
Văn bản: ```{contexts}```
Câu hỏi: ```{question}```
Trả lời: [/INST]''',
            "lang": "vi"
        }
        resp = requests.post(self.url, headers = self.header, data=json.dumps(info))
        print(resp)
        data = json.loads(resp.content)
        return data['answer']
''' Yeu cau

Văn bản:
Câu hỏi:
Trả lời:

Văn bản:
Câu hỏi:
Trả lời: ''' 
if __name__ == '__name__':
    pass