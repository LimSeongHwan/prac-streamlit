import streamlit as st
from dotenv import load_dotenv
from prac_langgraph import get_result

st.set_page_config(page_title="데이터 추출 챗봇", page_icon="🤖")

st.title("🤖 데이터 추출 챗봇")
st.caption("데이터를 모두 뽑아드립니다!")

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):

        ai_response = get_result(user_question)
        content_in_response = ai_response["chatbot"][0].content
        with st.chat_message("ai"):
            ai_message_chatbot = st.write(content_in_response)
            st.session_state.message_list.append({"role": "ai", "content": content_in_response})
