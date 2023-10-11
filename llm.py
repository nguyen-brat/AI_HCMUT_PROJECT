import requests
import json
import streamlit as st

class LlmInference:
    def __init__(
            self,
            url = st.secrets['API_KEY'],
            header = {"Content-Type": "application/json"},
    ):
        self.url = url
        self.header = header

    def __call__(self, question, contexts):
        info = {
            "prompt": f'''<s>[INST] <<SYS>>Trả lời câu hỏi đưa ra dựa vào văn bản được cung cấp nếu bạn không tìm thấy thông tin trong văn bản hay trả lời không có thông tin liên quan được tìm thấy hãy<</SYS>>
            văn bản:{contexts}
            câu hỏi:{question}
            [/INST]''',
            "lang": "vi"
        }
        resp = requests.post(self.url, headers = self.header, data=json.dumps(info))
        data = json.loads(resp.content)
        return data['answer']
    
if __name__ == '__name__':
    pass