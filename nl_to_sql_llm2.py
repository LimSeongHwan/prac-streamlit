from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from config_db import answer_examples

store = {}


class Nl_to_Sql:

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def get_retriever(self):
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        # embedding = UpstageEmbeddings(model="solar-embedding-1-large")
        index_name = 'tax-index'
        # index_name = 'tax-upstage-index'
        database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
        retriever = database.as_retriever(search_kwargs={'k': 4})
        return retriever

    def get_history_retriever(self):
        llm = self.get_llm()
        retriever = self.get_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        return history_aware_retriever


    def get_llm(model='gpt-4o'):
        llm = ChatOpenAI(model=model)
        return llm


    def get_dictionary_chain(self):
        dictionary = ["사람을 나타내는 표현 -> 거주자"]
        llm = self.get_llm()
        prompt = ChatPromptTemplate.from_template(f"""
            사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
            만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
            그런 경우에는 질문만 리턴해주세요
            사전: {dictionary}
            
            질문: {{question}}
        """)

        dictionary_chain = prompt | llm | StrOutputParser()

        return dictionary_chain


    def get_rag_chain(self):
        llm = self.get_llm()
        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{answer}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=answer_examples,
        )
        system_prompt = (
            "당신은 아웃소싱(Outsourcing) 업체선정 전문가입니다. 사용자의 업무 아웃소싱(Outsourcing)을 위한 업체 선정에 관한 질문에 답변해주세요"
            "아래에 제공된 문서를 활용해서 답변해주시고"
            "답변을 알 수 없다면 모른다고 답변해주시고 제공된 데이터 안에 존재 하지 않는 업체는 답변하지 마세요"
            "답변을 제공할 때는 아웃소싱 엑셀 파일에 따르면 이라고 시작하면서 답변해주시고"
            "제공된 데이터 안에서 사용자의 질문에 맞는 모든 업체를 답변해주시고, 제공된 데이터 안에서 최대한 업체의 모든 정보를 담아주세요"
            "그리고 가독성을 위해서 줄바꿈을 많이 활용해주세요"
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                few_shot_prompt,
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = self.get_history_retriever()
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        ).pick('answer')

        return conversational_rag_chain


    def get_ai_response(self, user_message):

        dictionary_chain = self.get_dictionary_chain()
        # print(dictionary_chain.invoke({"question" : user_message}))
        rag_chain = self.get_rag_chain()
        tax_chain = {"input": dictionary_chain} | rag_chain
        ai_response = tax_chain.stream(
            {
                "question": user_message
            },
            config={
                "configurable": {"session_id": "abc123"}
            },
        )

        return ai_response