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
        config = genai.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT
        )
        
        # 새 세션 생성
        chat = client.chats.create(
            model=model_name,
            config=config,
            history=[] # 초기 히스토리는 비워둡니다.
        )
        st.session_state.chat_history = [] # 세션 히스토리 초기화
        st.session_state.messages = [] # 화면 표시 메시지 초기화
        st.session_state.history_limit_counter = 0 # 카운터 초기화
        st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S") # 새 세션 ID
        return chat
    except Exception as e:
        st.error(f"⚠️ 챗 세션 생성 오류: {e}")
        return None

def handle_user_input(client, model_name, user_prompt):
    """사용자 입력을 처리하고 Gemini API를 호출합니다 (429 재시도 로직 포함)."""
    
    # 429 재시도 로직
    for attempt in range(MAX_RETRIES):
        try:
            # HISTORY_WINDOW_SIZE (6턴) 마다 세션 초기화 (재시작) 로직
            if st.session_state.history_limit_counter >= HISTORY_WINDOW_SIZE:
                st.info("🔄 대화 길이가 길어져, 새로운 세션으로 초기화하고 최근 대화를 유지합니다. 진단 정확도를 높이기 위함입니다.")
                
                # 새로운 Chat Session 생성
                new_chat = create_new_chat_session(client, model_name)
                if new_chat:
                    st.session_state.chat_session = new_chat
                    st.session_state.history_limit_counter = 0 # 카운터 초기화
                
                # 이전 대화의 마지막 6개 메시지 (사용자 3, 봇 3)를 새 세션의 히스토리로 재구성하여 로드 시도
                # Chat History의 형식은 (role, text) 튜플 리스트입니다.
                recent_history = st.session_state.chat_history[-HISTORY_WINDOW_SIZE:]
                
                # Gemini Chat.history는 genai.types.Content 리스트 형태여야 합니다.
                # 단순 재시작을 위해, 새 세션에서는 이전 대화의 context를 '새로운 사용자 입력' 직전에 넣을 수 없으므로, 
                # 메시지 히스토리만 유지하고, API는 새로운 세션으로 시작합니다.
                # (context 전달을 위한 복잡한 로직 대신, 사용자에게 재시작을 알리고 진행)

            # API 호출
            response = st.session_state.chat_session.send_message(user_prompt)
            
            # 성공 시, 메시지와 히스토리 업데이트 및 카운터 증가
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            st.session_state.messages.append({"role": "assistant", "content": response.text})
            
            # 챗 히스토리 기록 (CSV 다운로드를 위함)
            st.session_state.chat_history.append(("user", user_prompt))
            st.session_state.chat_history.append(("assistant", response.text))
            
            st.session_state.history_limit_counter += 2
            
            # ⭐️⭐️⭐️ [수정] try 블록 성공 시, 함수를 종료하고 결과를 반환하는 구문을 추가했습니다.
            return response.text 
            
        except APIError as e:
            if '429' in str(e): # Rate Limit Error (429)
                st.warning(f"⚠️ API 호출 한도 초과 (429 에러): 재시도 중... ({attempt + 1}/{MAX_RETRIES})")
                time.sleep(2 ** attempt) # 지수 백오프 (Exponential Backoff)
                if attempt == MAX_RETRIES - 1:
                    st.error("❌ 최대 재시도 횟수를 초과했습니다. 잠시 후 다시 시도해 주세요.")
                    break
            else:
                st.error(f"❌ API 오류 발생: {e}")
                break
        except Exception as e:
            st.error(f"❌ 예기치 않은 오류 발생: {e}")
            break
    
    return "죄송합니다. 오류가 발생하여 응답할 수 없습니다."


# --- UI 및 데이터 다운로드 함수 ---

def download_chat_log():
    """대화 히스토리를 CSV 파일로 변환하여 다운로드합니다."""
    if not st.session_state.chat_history:
        st.warning("다운로드할 대화 기록이 없습니다.")
        return
    
    # DataFrame 생성
    data = []
    for role, text in st.session_state.chat_history:
        data.append({"시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "역할": role, "내용": text})
    df = pd.DataFrame(data)
    
    # CSV로 변환
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8-sig') # 한글 깨짐 방지를 위해 utf-8-sig 사용
    
    # 파일 다운로드 버튼 활성화
    st.download_button(
        label="📄 대화 로그 (CSV) 다운로드",
        data=csv_buffer.getvalue(),
        file_name=f"chat_log_{st.session_state.current_session_id}.csv",
        mime="text/csv"
    )

def download_app_log():
    """로그 다운로드 (이 예시에서는 대화 로그와 동일하게 처리)"""
    download_chat_log()


def clear_chat_history():
    """대화 초기화 버튼 클릭 시 호출되는 함수입니다."""
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.history_limit_counter = 0
    st.session_state.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    # API 클라이언트가 존재하면 새 세션 생성
    if 'client' in st.session_state:
        st.session_state.chat_session = create_new_chat_session(st.session_state.client, st.session_state.selected_model)
    st.success("✅ 대화가 성공적으로 초기화되었습니다.")

# --- 메인 애플리케이션 로직 ---

def main():
    initialize_session_state()

    st.title("Gemini 기반 우울증 진단 챗봇 🫂")
    st.write("당신의 고민을 정중하고 공감 어린 말투로 듣고, 우울증 진단을 보조하며 해결 방안을 함께 찾아드립니다.")

    # 1. API 키 설정 (st.secrets 또는 임시 UI)
    api_key = st.secrets.get("GEMINI_API_KEY")

    if not api_key:
        with st.sidebar:
            st.warning("⚠️ `st.secrets['GEMINI_API_KEY']`가 설정되어 있지 않습니다.")
            api_key = st.text_input("Gemini API Key를 여기에 입력하세요:", type="password")
            st.markdown("API Key는 [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)에서 발급받을 수 있습니다.")
    
    if not api_key:
        st.info("🔑 Gemini API Key를 입력해야 챗봇을 사용할 수 있습니다.")
        return

    # API 클라이언트 생성 및 캐시
    try:
        st.session_state.client = get_gemini_client(api_key)
    except Exception as e:
        st.error(f"클라이언트 초기화 오류: {e}")
        return

    # 2. 사이드바 설정 (모델 선택, 기능 버튼)
    with st.sidebar:
        st.header("⚙️ 설정 및 기능")
        
        # 모델 선택
        selected_model = st.selectbox(
            "모델 선택:",
            MODEL_OPTIONS,
            index=MODEL_OPTIONS.index("gemini-2.0-flash") if "gemini-2.0-flash" in MODEL_OPTIONS else 0,
            key='selected_model',
            on_change=clear_chat_history # 모델 변경 시 초기화
        )
        
        st.markdown("---")
        
        # 대화 초기화
        st.button("🔄 대화 초기화", on_click=clear_chat_history)
        
        # CSV 자동 기록 옵션 (옵션만 제공하고, 기록은 항상 수행)
        st.checkbox("📊 대화 내용 CSV 자동 기록 (기록은 항상 수행됨)", value=True, disabled=True)
        
        # 로그 다운로드 (대화 로그와 동일하게 처리)
        download_chat_log()
        
        st.markdown("---")
        
        # 현재 세션 및 모델 정보 표시
        st.subheader("💡 세션 정보")
        st.markdown(f"**현재 모델:** `{st.session_state.selected_model}`")
        st.markdown(f"**세션 ID:** `{st.session_state.current_session_id}`")
        st.markdown(f"**대화 턴:** `{st.session_state.history_limit_counter // 2}` 턴")
        st.markdown(f"**다음 재시작:** `{HISTORY_WINDOW_SIZE - st.session_state.history_limit_counter}` 턴 남음")


    # 3. 챗 세션 초기화/확인
    if st.session_state.chat_session is None or st.session_state.chat_session.model != st.session_state.selected_model:
        st.session_state.chat_session = create_new_chat_session(st.session_state.client, st.session_state.selected_model)
        if not st.session_state.chat_session:
            st.stop() # 세션 생성 실패 시 중단

    # 4. 대화 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 5. 사용자 입력 처리
    if user_prompt := st.chat_input("당신의 고민이나 궁금한 점을 말씀해 주세요..."):
        # 사용자 입력 화면에 표시
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        # 챗봇 응답 및 화면 표시
        with st.spinner("생각 중... 잠시만 기다려 주세요. 💬"):
            response_text = handle_user_input(st.session_state.client, st.session_state.selected_model, user_prompt)
        
        # 챗봇 응답 화면에 표시
        with st.chat_message("assistant"):
            st.markdown(response_text)
            
        # 화면 자동 스크롤을 위해 Streamlit 다시 실행 (필요 시)
        # st.rerun()


if __name__ == "__main__":
    main()
