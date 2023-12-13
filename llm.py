import requests
import json
import streamlit as st

class LlmInference:
    def __init__(
            self,
            url,
            header,
    ):
        self.url = url
        self.header = header

    def __call__(self, question, contexts):
        contexts = ' '.join(contexts.split()[:600])
#         info = {
#             "inputs": f'''[INST] <<SYS>> Trả lời câu hỏi đưa ra dựa vào văn bản được cung cấp. Nếu không có dữ liệu liên quan trong văn bản hãy trả lời: "Không có tài liệu liên quan" <</SYS>>
# Văn bản: ```{contexts}```
# Câu hỏi: ```{question}```
# Trả lời: [/INST]''',
#             "parameters": {"max_new_tokens":100}
#         }

        print(contexts)
        print('-------------------------')
        info = {
#             "inputs": f'''[INST] <<SYS>> Trả lời câu hỏi đưa ra dựa vào văn bản được cung cấp <</SYS>>
# Văn bản: Tiền thân của Trường Đại học Bách khoa là Trung tâm Kỹ thuật Quốc gia được thành lập từ năm 1957
# Câu hỏi: Trường đại học Bách Khoa thành lập vào năm nào
# Trả lời: [/INST]
# Trường đại học Bách Khoa thành lập vào năm 1957
# [INST] Văn bản: {contexts}
# Câu hỏi: {question}
# Trả lời: [/INST]''',
            "inputs": f'''[INST] <<SYS>> Trả lời câu hỏi đưa ra dựa vào văn bản về trường đại học Bách Khoa Hồ Chí Minh được cung cấp <</SYS>>
Văn bản: ```{contexts}```
Câu hỏi: ```{question}```
Trả lời: [/INST]''',
            "parameters": {"max_new_tokens":300, 'truncate':1024, 'temperature':1, 'top_k':3, 'top_p':0.9, 'repetition_penalty':1.1, 'do_sample':True}
        }
        resp = requests.post("https://www.ura.hcmut.edu.vn/ura-llama/generate", headers = {"Content-Type": "application/json", "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2xvZ2luIjoiTmd1eWVuIn0._-roV2s_je9vyZt2wa-pxNSK7SaPH1qh1XprSAH8ekE"}, data=json.dumps(info))
        data = json.loads(resp.content)
        return data['generated_text']


        # try:
        #     resp = requests.post(self.url, headers = self.header, data=json.dumps(info))
        #     print(resp)
        #     # print(self.url)
        #     # print(self.header)
        #     data = json.loads(resp.content)
        #     return data['generated_text']
        # except:
        #     print('response error')
        #     return None
''' Yeu cau

Văn bản:
Câu hỏi:
Trả lời:

Văn bản:
Câu hỏi:
Trả lời: ''' 
if __name__ == '__name__':
    pass