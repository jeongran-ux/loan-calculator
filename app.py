import math
from dataclasses import dataclass
from datetime import date
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.family"] = "Malgun Gothic"   # 윈도우 한글 폰트
mpl.rcParams["axes.unicode_minus"] = False      # 마이너스(-) 깨짐 방지
import streamlit as st


# =============================
# 금융 계산 로직
# =============================

def pmt_monthly(principal: float, annual_rate: float, months: int) -> float:
    """원리금균등 월 납입액 (PMT)"""
    if months <= 0:
        return 0
    r = annual_rate / 12
    if r == 0:
        return principal / months
    return principal * r / (1 - (1 + r) ** (-months))


def add_months(d: date, m: int) -> date:
    y = d.year + (d.month - 1 + m) // 12
    month = (d.month - 1 + m) % 12 + 1
    day = min(d.day, [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30,
                      31, 31, 30, 31, 30, 31][month - 1])
    return date(y, month, day)


@dataclass
class Inputs:
    balance: float
    annual_rate: float
    months: int
    method: str
    extra: float
    start_date: date
    equal_mode: str


def build_schedule(inp: Inputs) -> pd.DataFrame:
    bal = inp.balance
    r = inp.annual_rate / 12
    n = inp.months

    rows = []

    # 원리금균등 자동 계산
    fixed_payment = None
    if inp.method == "원리금균등":
        fixed_payment = pmt_monthly(bal, inp.annual_rate, n)

    original_balance = bal
    fixed_principal = original_balance / n

    for i in range(1, n + 1):
        if bal <= 0:
            break

        interest = bal * r

        if inp.method == "원리금균등":
            principal = max(0, fixed_payment - interest)

        elif inp.method == "원금균등":
            if inp.equal_mode == "초기잔액기준":
                principal = fixed_principal
            else:
                principal = bal / (n - i + 1)
            principal = min(bal, principal)

        elif inp.method == "만기일시상환":
            principal = bal if i == n else 0

        else:
            principal = 0

        extra_principal = min(inp.extra, bal - principal)
        payment = interest + principal + extra_principal
        end_balance = max(0, bal - principal - extra_principal)

        rows.append({
            "회차": i,
            "납부월": add_months(inp.start_date, i - 1),
            "기초잔액": round(bal),
            "이자": round(interest),
            "정기원금": round(principal),
            "추가원금": round(extra_principal),
            "총납입액": round(payment),
            "기말잔액": round(end_balance)
        })

        bal = end_balance

    return pd.DataFrame(rows)


# =============================
# Streamlit UI
# =============================

from datetime import date
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="대출 상환 계산기", layout="wide")
st.title("대출 상환 계산기")

with st.sidebar:
    st.header("입력값")

    balance = st.number_input("현재 대출잔액(원)", min_value=0, value=9_076_838, step=10000)
    rate = st.number_input("연이자율(%)", min_value=0.0, value=15.9, step=0.1)

    method = st.selectbox("상환방식", ["원리금균등", "원금균등", "만기일시상환"])
    extra = st.number_input("매월 추가 원금상환(원)", min_value=0, value=500_000, step=10000)

    loan_start_date = st.date_input("대출일", value=date(2025, 6, 2))
    loan_end_date = st.date_input("대출 만기일", value=date(2030, 6, 2))

    equal_mode = "초기잔액기준"
    if method == "원금균등":
        equal_mode = st.radio("원금균등 계산 기준", ["초기잔액기준", "현재잔액기준"])

# -----------------------------
# 남은 상환개월수 자동 계산
# -----------------------------
today = date.today()
base_date = max(today, loan_start_date)

rd = relativedelta(loan_end_date, base_date)
months = rd.years * 12 + rd.months
months = max(months, 1)  # 최소 1개월 보장

# (선택) 사이드바에 표시
with st.sidebar:
    st.caption(f"▶ 오늘 기준 남은 상환기간: {months}개월")

# -----------------------------
# 상환 시작월 = 기준월(오늘 or 대출일 중 더 늦은 날)
# -----------------------------
start_date = base_date

inp = Inputs(
    balance=balance,
    annual_rate=rate / 100,
    months=months,
    method=method,
    extra=extra,
    start_date=start_date,
    equal_mode=equal_mode
)

df = build_schedule(inp)

# -----------------------------
# 표시용 DataFrame (₩ + 콤마)
# -----------------------------
money_cols = ["기초잔액", "이자", "정기원금", "추가원금", "총납입액", "기말잔액"]

df_view = df.copy()
for col in money_cols:
    df_view[col] = df_view[col].apply(lambda x: f"₩{x:,.0f}")

# =============================
# 요약 영역
# =============================

if not df.empty:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("완납까지 개월 수", f"{df.iloc[-1]['회차']}개월")

    with col2:
        st.metric("완납 예상월", df.iloc[-1]["납부월"].strftime("%Y-%m"))

    with col3:
        st.metric("총 이자", f"{df['이자'].sum():,}원")

    with col4:
        if method == "원리금균등":
            st.metric("정기상환액(자동)", f"{pmt_monthly(balance, rate/100, months):,.0f}원")
        else:
            st.metric("정기상환액", "해당 없음")

st.divider()

# =============================
# 테이블
# =============================

st.subheader("월별 상환 스케줄")
st.dataframe(df_view, use_container_width=True, height=420)

csv = df.to_csv(index=False).encode("utf-8-sig")
st.download_button("CSV 다운로드", csv, "loan_schedule.csv", "text/csv")

st.divider()

# =============================
# 그래프
# =============================

c1, c2 = st.columns(2)

with c1:
    st.caption("추가상환을 반영했을 때, 남은 대출원금이 어떻게 줄어드는지 보여줍니다.")
    fig1 = plt.figure()
    plt.plot(df["회차"], df["기말잔액"])
    plt.title("남은 대출원금 변화")
    plt.xlabel("월")
    plt.ylabel("금액(원)")
    plt.ticklabel_format(style="plain")
    st.pyplot(fig1)

with c2:
    st.caption("원금 감소에 따라 매달 부담하는 이자가 어떻게 줄어드는지 확인할 수 있습니다.")
    fig2 = plt.figure()
    plt.plot(df["회차"], df["이자"])
    plt.title("월 이자 부담 변화")
    plt.xlabel("월")
    plt.ylabel("금액(원)")
    plt.ticklabel_format(style="plain")
    st.pyplot(fig2)