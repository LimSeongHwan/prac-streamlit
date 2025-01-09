# API 키를 환경변수로 관리하기 위한 설정 파일
from dotenv import load_dotenv
from typing import Annotated, List, Dict
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.runnables import RunnableConfig
from nl_to_sql_llm import Nl_to_Sql
from langchain_teddynote.tools import GoogleNews

news_tool = GoogleNews()

# API 키 정보 로드
load_dotenv()

########## 1. 상태 정의 ##########
# 상태 정의
class State(TypedDict):
    # 메시지 목록 주석 추가
    messages: Annotated[list, add_messages]
    # dummy_data: Annotated[str, "dummy"]


########## 2. 도구 정의 및 바인딩 ##########
# 도구 초기화

# def get_llm():
#     llm = ChatOpenAI(model="gpt-4o-mini")
#     return llm

def get_llm_with_tools(tool):
    # 도구와 LLM 결합
    llm = ChatOpenAI(model="gpt-4o-mini")
    llm_with_tools = llm.bind_tools(tool)
    return llm_with_tools

@tool
def nl_to_sql(query: str) -> List[Dict[str, str]]:
    """nl_to_sql"""
    nl_to_sql_ins = Nl_to_Sql()
    return nl_to_sql_ins.get_ai_response(str)

########## 3. 노드 추가 ##########
# 챗봇 함수 정의
def chatbot(state: State):
    nl_to_sql_ins = Nl_to_Sql()
    llm_with_tools = get_llm_with_tools(nl_to_sql_ins)
    # 메시지 호출 및 반환
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
        # "dummy_data": "[chatbot] 호출, dummy data",  # 테스트를 위하여 더미 데이터를 추가합니다.
    }

def create_graph():
    # 상태 그래프 생성
    graph_builder = StateGraph(State)

    # 챗봇 노드 추가
    graph_builder.add_node("chatbot", chatbot)

    # 도구 노드 생성 및 추가
    tool_node = ToolNode(tools=[nl_to_sql])

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

    ########## 5. 그래프 컴파일 ##########
    return graph_builder.compile()

########## 6. 그래프 시각화 ##########
# 그래프 시각화
# visualize_graph(graph)

def result(question):
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

    for event in graph.stream(input=input, config=config):
        for key, value in event.items():
            print(f"\n[ {key} ]\n")
            # value 에 messages 가 존재하는 경우
            if "messages" in value:
                messages = value["messages"]
                # 가장 최근 메시지 1개만 출력합니다.
                value["messages"][-1].pretty_print()