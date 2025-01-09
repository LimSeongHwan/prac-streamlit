# API 키를 환경변수로 관리하기 위한 설정 파일
import pandas
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from llm_db import nlToSql
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.tools import GoogleNews
from typing import List, Dict


# API 키 정보 로드
load_dotenv()
# Node 단계별 상태 기억
memory = MemorySaver()

# 상태 정의
class State(TypedDict):
    # 메시지 목록 주석 추가
    messages: Annotated[list, add_messages]

def get_llm_with_tools(tools):
    llm = ChatOpenAI(model="gpt-4o-mini")
    # 도구와 LLM 결합
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools

@tool
def nl_to_sql(question: str):
    """A tool that converts natural language into queries"""
    nl_to_sql_ins = nlToSql()
    return nl_to_sql_ins.get_db_ai_response(question)

@tool
def search_news(query: str) -> List[Dict[str, str]]:
    """Search Google News by input keyword"""
    news_tool = GoogleNews()
    return news_tool.search_by_keyword(query, k=5)

# 챗봇 함수 정의
def chatbot(state: State):
    llm_with_tools = get_llm_with_tools([nl_to_sql, search_news])
    # 메시지 호출 및 반환
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
    }

def create_graph():
    # 상태 그래프 생성
    graph_builder = StateGraph(State)

    # 챗봇 노드 추가
    graph_builder.add_node("chatbot", chatbot)

    # 도구 노드 생성 및 추가
    tool_node = ToolNode(tools=[nl_to_sql, search_news])

    # 도구 노드 추가
    graph_builder.add_node("tools", tool_node)

    # 조건부 엣지
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )

    ########## 4. 엣지 추가 ##########
    # tools > chatbot
    graph_builder.add_edge("tools", "chatbot")

    # START > chatbot
    graph_builder.add_edge(START, "chatbot")

    # chatbot > END
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile(checkpointer=memory)

    return graph


def get_result(question):
    graph = create_graph()
    # 초기 입력 상태를 정의
    # input = State(dummy_data="테스트 문자열", messages=[("user", question)])
    input = State(messages=[("user", question)])

    # config 설정
    config = RunnableConfig(
        recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
        configurable={"thread_id": "1"},  # 스레드 ID 설정
        tags=["my-tag"],  # Tag
    )

    response_dict = {}

    for event in graph.stream(input=input, config=config):
        for key, value in event.items():
            # value 에 messages 가 존재하는 경우
            if "messages" in value:
                messages = value["messages"]
                response_dict[key] = messages

    return response_dict

