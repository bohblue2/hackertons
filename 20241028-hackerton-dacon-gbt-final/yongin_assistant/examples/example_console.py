import streamlit as st
from datetime import datetime

# 페이지 기본 설정
st.set_page_config(page_title="답변 시스템", layout="wide")

# 사이드바 
with st.sidebar:
    st.header("국토교통부")
    st.button("경기도 이천시")
    st.button("개임화 관리 위원회")

# 메인 영역
st.header("전체 조회", divider="red")

key = 1

# 답변 카드를 만드는 함수
def create_answer_card(title, content="", date=None, has_buttons=True):
    global key
    with st.container():
        st.markdown(f"### {title}")
        if date:
            st.text(f"접수 일자: {date}")
        
        # 빈 컨텐츠 영역
        st.text_area("", key=str(key), value=content, height=100, disabled=True)
        
        # 버튼 영역
        if has_buttons:
            cols = st.columns([1, 1, 8])  # 버튼 정렬을 위한 컬럼 설정
            with cols[0]:
                st.button("내용 수정", key=f"edit_{title}")
            with cols[1]:
                st.button("답변 등록", key=f"reply_{title}")
        
        key += 1

# 여러 답변 카드 생성
create_answer_card("(민원인 민원 내용1)", date="2024.10.26")
create_answer_card("(답당 공무원 추천 답변 내용)")
create_answer_card("(민원인 민원 내용2)", date="2024.10.26")
create_answer_card("(답변 내용)", has_buttons=False)

# 하단 버튼
cols = st.columns([1, 1, 8])
with cols[0]:
    st.button("더보기")
with cols[1]:
    st.button("전체 목록 보기")

# 스타일 적용
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        background-color: #4A4A4A;
        color: white;
    }
    .stTextArea textarea {
        background-color: #2E2E2E;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)