import pandas as pd
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv
from config_db import answer_examples
from connect_db import connect_db

load_dotenv()

store = {}

class nlToSql:

    llm = ChatOpenAI(model="gpt-4o-mini")

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    def get_retriever(self):
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        # embedding = UpstageEmbeddings(model="solar-embedding-1-large")
        # index_name = 'excel-test-idx'
        index_name = 'tax-index'
        database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
        retriever = database.as_retriever(search_kwargs={'k': 4})
        return retriever

    # def get_history_retriever(self):
    #     retriever = self.get_retriever()
    #
    #     contextualize_q_system_prompt = (
    #         "Given a chat history and the latest user question "
    #         "which might reference context in the chat history, "
    #         "formulate a standalone question which can be understood "
    #         "without the chat history. Do NOT answer the question, "
    #         "just reformulate it if needed and otherwise return it as is."
    #     )
    #
    #     contextualize_q_prompt = ChatPromptTemplate.from_messages(
    #         [
    #             ("system", contextualize_q_system_prompt),
    #             MessagesPlaceholder("chat_history"),
    #             ("human", "{input}"),
    #         ]
    #     )
    #
    #     history_aware_retriever = create_history_aware_retriever(
    #         self.llm, retriever, contextualize_q_prompt
    #     )
    #     return history_aware_retriever


    # def get_llm(model='gpt-4o'):
    #     llm = ChatOpenAI(model=model)
    #     return llm

    def nl_sql_chain(self):
        dictionary = [
            "database table list -> db_server, db_service",
            "database table explain -> db_service : 데이터베이스 서비스에 대한 정보를 담고 있는 테이블이며, 컬럼은 id, created_at, updated_at, created_by, updated_by, use_flag, company_code, contract_id, cpu, data_center, database_name, db_user, disk_type, disk_vol, dpm_resource_group_id, ha_type, mem, node_count, port, product_id, service_name, status, engine_id로 이루어져 있다",
            "database table explain -> db_server : 데이터베이스 서비스에 대한 정보를 담고 있는 테이블이며 db_service와 1:N 관계를 가지고 있는 테이블이다, db_service에 db_server가 포함되는 구조이다. 컬럼은 id, created_at, updated_at, created_by, updated_by, use_flag, backup_ip, bck_disk_vol, count_number, dpm_resource_group_id, management_ip, master_flag, replication_ip, resource_id, service_ip, type, vip, service_id로 이루어져 있다",
            "database column explain -> id : table의 pk 값, created_at : row의 생성일자, updated_at : row의 수정일자, created_by : row를 생성한 사람, updated_by : row를 수정한 사람, use_flag : 사용여부, backup_ip : 백업망 ip, resource_id : db_prac 서버의 자원명, service_id : db_prac 서버와 연결된 db_prac service의 id 값, data_center : db_prac service가 위치한 지역 (SL은 서울, YI는 용인), ha_type : 가용성 타입으로 signle, replication, cluster가 있다, mem : memory 크기, service_name : 서비스 이름"
        ]
        # llm = self.get_llm()
        prompt = ChatPromptTemplate.from_template(f"""
            "당신은 자연어를 sql문으로 변환하는 전문가입니다. 사용자의 질문을 우리가 가지고 있는 사전을 참고하여 sql문을 생성해주세요"
            "SQL문을 보고 실제로 존재하는 테이블이 맞는지 확인하고 존재하지 않는 테이블이라면 SQL을 다시 생성해주세요"
            "그리고 테이블 간의 조인 관계를 확인해주고, 테이블에 없는 컬럼을 추출하려고 하진 않는지, WHERE 조건문에 테이블에 없는 컬럼이 포함되어 있는지 확인하고 만약 있다면 수정해서 다시 생성 해주세요"
            "반환할 때는 ```sql 태그 없이 그 어떤 태그도 적용하지말고 오직 SQL문만 TEXT로 리턴해주세요"
            "데이터 추출이 아닌 경우에는 사용자의 질문을 그대로 리턴해주세요"
            
            사전 : {dictionary}
            질문: {{question}}
        """)

        dictionary_chain = prompt | self.llm | StrOutputParser()

        return dictionary_chain


    def get_rag_chain(self):

        retriever = self.get_retriever()

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
            "SQL문을 보고 문법적 오류가 있는지 확인해주세요"
            "SQL문을 보고 제공한 문서를 참고해서 실제 존재하는 컬럼과 테이블 인지 확인해주세요"
            "오류가 없을 경우 SQL문을 그대로 반환하고, 오류가 있을 경우 오류를 수정해서 반환해주세요"
            "반환할 때는 ```sql 태그 없이 그 어떤 태그도 적용하지말고 오직 SQL문만 TEXT로 리턴해주세요"
            "만약 사용자의 질문이 SQL이 아닐 경우에는 질문에 대해 대답해주시면 됩니다"
            "\n\n"
            "문서: {context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                few_shot_prompt,
                # MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # history_aware_retriever = self.get_history_retriever()
        #
        # question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        #
        # rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # conversational_rag_chain = RunnableWithMessageHistory(
        #     rag_chain,
        #     self.get_session_history,
        #     input_messages_key="input",
        #     history_messages_key="chat_history",
        #     output_messages_key="answer",
        # ).pick('answer')

        history_aware_retriever = self.get_history_retriever()

        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=retriever, chain_type_kwargs={"prompt" : qa_prompt})

        return rag_chain


    def get_db_ai_response(self, user_message):

        rag_chain = self.get_rag_chain()

        nl_sql_chain = self.nl_sql_chain()

        tax_chain = {"query": rag_chain} | nl_sql_chain
        # tax_chain = {"input": nl_sql_chain}

        ai_message = tax_chain.invoke(
            {
                "question": user_message
            },
            config = {
                "configurable" :
                    {
                     "session_id": "abc123"
                    }
            }
        )

        if ai_message.startswith("SELECT"):
            return pd.DataFrame(connect_db(ai_message))

        return ai_message

abc = nlToSql()
print(abc.get_db_ai_response("db 서비스 데이터 뽑아줘"))