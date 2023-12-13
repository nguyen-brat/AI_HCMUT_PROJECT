import streamlit as st
import os
from run import Inference
import requests
import json

email = ''
password = ''

URL = "https://www.ura.hcmut.edu.vn/ura-llama/login"
headers  = {"Content-Type": "application/json"}

info = {
    "email": email,
    "password": password,
}

# resp = requests.post(URL, headers = headers, data=json.dumps(info))
# data= json.loads(resp.content)
# token = data['data']['token']

resp = ''
data= ''
token = ''

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.image(["logo.jpg"], width=120)
    inference = Inference(
        url='https://www.ura.hcmut.edu.vn/ura-llama/generate',
        header={"Content-Type": "application/json", "Authorization": token},
    )
    st.title('Chatbot Demo')

    header = st.container()
    with header:
        question = st.text_input("Nhập câu hỏi", placeholder="Điều kiện để tốt nghiệp là gì ?")
        button = st.button("Submit")
        if button and question != "":
            answer, information = inference(question)
            information = information + [""]*(3-len(information))
        else:
            answer, information = "", ["", "", ""]
        st.write('Câu trả lời là:')
        if answer != None:
            st.write(f'{answer}')
        else:
            st.write(f'Server LLM hiện tại bị lỗi !!!')
        st.header('Các tài liệu liên quan được tìm thấy:', divider='rainbow')
        first_doc, second_doc, third_doc = st.columns(3)
        with first_doc:
            st.write(f'{information[0]}')
        with second_doc:
            st.write(f'{information[1]}')
        with third_doc:
            st.write(f'{information[2]}')