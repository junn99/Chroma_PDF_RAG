# streamlit
import streamlit as st
# langchain
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.runnable import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages.chat import ChatMessage
# etc..
import time
import os
from dotenv import load_dotenv
from operator import itemgetter

# langsmith 추적
langchain_api_key_env = os.getenv("LANGCHAIN_API_KEY","")


# API KEY 로드
load_dotenv()

# 캐시 디렉토리 생성
    # 앞에 .이 있으면 숨김폴더로 운용
if not os.path.exists(".cache"):
    os.mkdir(".cache")
# 파일 업로드 전용 폴더(임시로)
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("Insurance with GROQ")

# 사이드바 
with st.sidebar:
    clear_button = st.button("대화 초기화")
    
    selected_model = st.selectbox(
        "사용할 모델을 선택해 주세요.",
        ("llama-3.1-70b-versatile","llama-3.2-90b-text-preview","llama3-70b-8192","gemma-7b-it","gemma2-9b-it"),
        index=0
    )
    st.warning('답변이 제한되면, 다른 모델로 바꿔주세요.',icon="⚠️")

    # 세션 ID 지정 : 다른 카톡방 느낌
    session_id = st.text_input("세션 ID를 입력하세요.","abc123")



# ChatGroq 전용 세션 상태 초기화
if "groq_messages" not in st.session_state:
    st.session_state["groq_messages"] = []

# if "groq_chain" not in st.session_state:
#     st.session_state["groq_chain"] = None  # ChatGroq 모델 초기화

if "groq_store" not in st.session_state:
    st.session_state["groq_store"] = {}


# 대화기록 저장 함수(세션에 새로운 메시지 추가)
def add_message(role, message):
    st.session_state["groq_messages"].append(ChatMessage(role=role, content=message))



DB_PATH = "/home/jun/my_project/langchain_tutorial/Langcahin_Tutoraial/vectorDB/insurance"
# 파일이 업로드 되었을 때
@st.cache_resource(show_spinner="기존의 DB를 가져오는 중입니다...!") # 캐싱된 파일을 사용하는 데코레이터
def load_chromaDB(DB_PATH):

    # 임베딩 모델 정의
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device":"cpu"}
    encode_kwargs = {"normalize_embeddings":True}
    embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs,encode_kwargs=encode_kwargs
    
    )
    # 디스크에서 문서를 로드합니다.
    persist_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings,
        collection_name="insurance_chroma_db",
    ) 

    retriever = persist_db.as_retriever()

    return retriever


# 세션 ID를 기반으로 세션 기록을 가져오는 함수 : 캐싱된 걸 가져오는 걸로 바꿔줌
def get_session_history(session_ids):
    if session_ids not in st.session_state["groq_store"]:
        st.session_state["groq_store"][session_ids] = ChatMessageHistory()
    return st.session_state["groq_store"][session_ids] # 해당 세션ID에 대한 기록 반환


# 체인 생성
def create_chain(model_name=selected_model):
    # 프롬프트 정의 수정
    prompt = ChatPromptTemplate.from_template(
        """You are an assistant for question & answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that don't know.
        Answer in korean.

        #Context:
        {context}

        #Previous Chat History:
        {chat_history}

        #Question:
        {question}

        #Answer:
        """
    )

    llm = ChatGroq(model=model_name, temperature=0.7)

    retriever = load_chromaDB(DB_PATH)

    # Chain 생성
    chain = (
        {
            "context": itemgetter("question") | retriever, 
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history")
        } 
        | prompt 
        | llm 
        | StrOutputParser()
    )

    rag_chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return rag_chain_with_history




# 이전 대화 기록 출력 함수
def print_messages():
    for chat_message in st.session_state["groq_messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)



# 초기화 버튼이 눌리면... 대화기록 초기화 -> 그래서? 대화기록 출력하기 전에 위치시킴
if clear_button:
        st.session_state["groq_messages"] = [] # 빈리스트로 초기화

print_messages() # 호출 : 대화기록 출력


# 사용자 입력
user_input = st.chat_input("보험에 대해 물어보세요.")

# 경고 메시지를 띄우기 위한 빈 영엉
warning_message = st.empty()

if "groq_chain" not in st.session_state:
    st.session_state["groq_chain"] = create_chain(model_name=selected_model)

if user_input: # 사용자 입력(user_input)이 들어오면

    # 매번 가져오는 게 아닌, 세션에 저장된 체인 가져옴
    chain = st.session_state["groq_chain"]

    if chain is not None:
        response = chain.stream( # RunnablePassthrough 쓰면 이렇게 dict아닌, 값만 넣어줘야 함
            # 질문 입력
            {"question":user_input},
            # 세션 ID 기준으로 대화 기록
            config={"configurable":{"session_id":session_id}},
        ) 

        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력
            container = st.empty()
            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        

            # 대화기록 저장
            add_message("user",user_input)
            add_message("assistant",ai_answer)
    
    else:
        warning_message.error("DB를 연결해주세요.")


##################################
    # 토큰 사용량 띄우고싶은데, 쉽지 않다
