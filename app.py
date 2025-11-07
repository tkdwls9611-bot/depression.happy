import streamlit as st
import pandas as pd
import io
import time
from datetime import datetime
from google import genai
from google.genai.errors import APIError

# --- 환경 설정 및 상수 정의 ---
# Streamlit 페이지 설정
st.set_page_config(
    page_title="Gemini 기반 우울증 진단 챗봇 🫂",
    page_icon="🤖",
    layout="wide"
)

# 세션 상태 초기화 함수
def initialize_session_state():
    """챗봇의 세션 상태를 초기화합니다."""
    # st.session_state가 초기화될 때 모든 키를 명시적으로 설정합니다.
    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if 'history_limit_counter' not in st.session_state:
        st.session_state.history_limit_counter = 0
    # 선택된 모델도 초기화 시점에 설정하여 None 오류를 방지합니다.
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = "gemini-2.0-flash" 

# 시스템 프롬프트 (요청 스펙 반영)
SYSTEM_PROMPT = """
당신은 우울증 진단 및 심리 상담을 보조하는 AI 챗봇입니다.

1. 사용자는 자신이 우울증인지 궁금하거나, 심리적으로 힘든 사람들이 자신의 고민과 걱정들을 언급합니다. 당신은 **정중하고 공감 어린 말투**로 응답해야 합니다. 사용자의 불편 사항에 대해 깊이 공감하고, 그들의 감정을 존중하는 것이 **필수**입니다.

2. 사용자의 고민, 상황의 사항을 **구체적으로 정리**하여(무엇이/언제/어디서/어떻게) 수집하고, 다음 3가지 항목에 대해 진단합니다:
   가. 우울증이 맞는지 여부
   나. 우울증의 예상 단계 (예: 경미, 보통, 심각 등)
   다. 왜 우울증이 맞고, 이 단계인지에 대한 자세한 이유를 **전문적이지만 이해하기 쉽게** 설명합니다.

3. 마지막에는 고민 상황에 맞는 **해결 방안**을 찾아서 자세하게 설명해주시고, 심한 단계일 경우 **진심 어린 위로**와 함께 심리상담이 가능한 사이트(예: 정신건강복지센터, 한국상담심리학회 등)를 **바로 연결**해주는 형식(Markdown 링크)으로 안내합니다.
"""

# 모델 선택 옵션
MODEL_OPTIONS = [
    "gemini-2.5-flash", 
    "gemini-2.0-flash", 
    "gemini-2.5-pro", 
    "gemini-2.0-pro"
]

# 429 에러 재시도 설정
MAX_RETRIES = 3
HISTORY_WINDOW_SIZE = 6 # 6턴 (사용자 3, 봇 3) 유지 후 재시작 (요청 스펙: 최근 6턴 유지 후 재시작)

# --- Gemini API 설정 및 챗봇 세션 관리 함수 ---

@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key):
    """Gemini 클라이언트를 생성하고 캐시합니다."""
    return genai.Client(api_key=api_key)

def create_new_chat_session(client, model_name):
    """새로운 Gemini Chat 세션을 생성하고 초기화합니다."""
    try:
        # 시스템 프롬프트를 포함한 설정 구성
        config = genai.
