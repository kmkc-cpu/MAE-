import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

# 페이지 설정
st.set_page_config(page_title="SK하이닉스 악취 분석 시스템", layout="wide")
st.title("👃 AI 복합악취(OU) 예측 시스템")

# 사이드바 메뉴
menu = st.sidebar.selectbox("메뉴 선택", ["실시간 분석", "AI 모델 학습"])

# --- 메뉴 1: 실시간 분석 ---
if menu == "실시간 분석":
    st.header("🔍 시료 실시간 판독")
    
    if not os.path.exists('odor_model.pkl'):
        st.error("학습된 AI 모델이 없습니다. 먼저 'AI 모델 학습'을 진행해 주세요.")
    else:
        uploaded_file = st.file_uploader("분석할 엑셀 파일을 업로드하세요", type=['xlsx'])
        
        if uploaded_file is not None:
            df = pd.read_excel(uploaded_file)
            # 데이터 지문 추출 (180x13)
            sensor_data = df.iloc[:180, 2:15].values.flatten().reshape(1, -1)
            
            if st.button("분석 실행"):
                model = joblib.load('odor_model.pkl')
                prediction = model.predict(sensor_data)
                
                # 결과 표출
                st.success(f"분석 완료! 시료의 예상 복합악취 수치(OU)는 {prediction[0]:.0f} 입니다.")
                st.line_chart(df.iloc[:180, 2:15]) # 센서 변화 그래프도 시각화

# --- 메뉴 2: AI 모델 학습 ---
elif menu == "AI 모델 학습":
    st.header("🧠 AI 두뇌 학습 시키기")
    st.write("여러 개의 데이터를 업로드하여 AI의 정확도를 높일 수 있습니다.")
    
    train_files = st.file_uploader("학습용 엑셀 파일들을 선택하세요 (중복 가능)", type=['xlsx'], accept_multiple_files=True)
    
    if st.button("AI 학습 시작"):
        X, y = [], []
        for f in train_files:
            df = pd.read_excel(f)
            df.columns = [c.strip() for c in df.columns]
            sensor_data = df.iloc[:180, 2:15].values
            if sensor_data.shape == (180, 13):
                X.append(sensor_data.flatten())
                y.append(float(df['ou'].iloc[0]))
        
        if len(X) >= 2:
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(np.array(X), np.array(y))
            joblib.dump(model, 'odor_model.pkl')
            st.balloons()
            st.success(f"학습 성공! {len(X)}개의 시료 지문을 마스터했습니다.")
        else:
            st.warning("학습을 위해 최소 2개 이상의 정상 파일이 필요합니다.")