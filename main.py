import streamlit as st
from run import Inference


if __name__ == '__main__':
    inference = Inference()
    st.title('Chatbot Demo')

    header = st.container()
    with header:
        question = st.text_input('Câu hỏi')
        answer, information = inference(question)
        st.write('Câu trả lời là:')
        st.write(f'{answer}')
        st.header('Các tài liệu liên quan được tìm thấy:', divider='rainbow')
        first_doc, second_doc, third_doc = st.columns(3)
        with first_doc:
            st.write(f'{information[0]}')
        with second_doc:
            st.write(f'{information[1]}')
        with third_doc:
            st.write(f'{information[2]}')