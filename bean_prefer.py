import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# 모델 로드
c_model = joblib.load("./model/Kmeans_model.joblib")
rf_model = joblib.load("./model/모든 데이터 학습_best_rf_model.joblib")

# 데이터 로드
data = pd.read_csv("./model/원두와 5가지 지표.csv")
data.set_index("Bean", inplace=True)
cosine_sim = cosine_similarity(data)
cosine_sim_df = pd.DataFrame(cosine_sim, index=data.index, columns=data.index)

brand_names = ["TheVenti", "Mega", "Paik", "Starbucks", "Ediya", "Compose", "Twosome"]

if 'dislike_list' not in st.session_state:
    st.session_state.dislike_list = []

def evaluate_recommendations(base_bean, recommended_beans):
    liked_beans = []

    while True:
        st.write("\n추천 원두:")
        for i, bean in enumerate(recommended_beans, start=1):
            st.write(f"{i}. {bean}")

        user_feedback = {}
        for bean in recommended_beans:
            if bean not in liked_beans and bean not in st.session_state.dislike_list:
                feedback = st.radio(f"{bean}에 대해 평가해주세요", ["호", "불호"], key=bean)
                user_feedback[bean] = 1 if feedback == "호" else 2

        for bean, feedback in user_feedback.items():
            if feedback == 2 and bean not in st.session_state.dislike_list:
                st.session_state.dislike_list.append(bean)
            elif feedback == 1 and bean not in liked_beans:
                liked_beans.append(bean)

        if len(user_feedback) > 0 and all(feedback == 1 for feedback in user_feedback.values()):
            st.write("\n추천이 종료됩니다.")
            break

        all_candidates = cosine_sim_df[base_bean].sort_values(ascending=False).drop(
            st.session_state.dislike_list + liked_beans + brand_names + [base_bean], axis=0
        )
        additional_beans = list(all_candidates.head(3 - len(liked_beans)).index)
        recommended_beans = liked_beans + additional_beans

    return liked_beans

st.title("커피 원두 추천 시스템")

purchase_history = st.radio("원더룸에서 원두를 구입해 본 적이 있습니까?", ["예", "아니오"])

exclude_beans = ["TheVenti", "Mega", "Paik", "Starbucks", "Ediya", "Compose", "Twosome", "Ethiopia Yirgacheffe Kochere Washed"]

if purchase_history == "예":
    purchased_bean = st.selectbox("구입했던 원두를 선택해주세요", data.index)

    if st.button("추천 받기"):
        recommended_beans = list(
            cosine_sim_df[purchased_bean]
            .sort_values(ascending=False)
            .drop([purchased_bean] + brand_names + st.session_state.dislike_list, axis=0)
            .head(3).index
        )
        evaluate_recommendations(purchased_bean, recommended_beans)
else:
    sex = st.radio("성별을 선택하세요", ["남", "여"])
    age = st.slider("나이를 입력하세요", 18, 60, 25)
    is_student = st.radio("직업을 선택하세요", ["학생", "기타"])
    frequency = st.selectbox("커피를 마시는 빈도", ["매일", "주 5-6회", "주 3-4회", "주 2회", "주 1회 미만"])
    method = st.selectbox("커피 내리는 방법", ["에스프레소 머신", "핸드 드립", "커피메이커", "콜드브루"])
    coffee_type = st.selectbox("커피 타입", ["블랙", "우유 라떼", "시럽 커피", "설탕 커피"])
    flavor = st.selectbox("커피 풍미", ["고소한, 구운", "달콤, 설탕", "초콜릿", "과일", "꽃향"])

    if st.button("추천 카페 찾기"):
        x = [1 if sex == "남" else 0, age, 1 if is_student == "학생" else 0,
             9 if frequency == "매일" else 7 if frequency == "주 5-6회" else 5 if frequency == "주 3-4회" else 3 if frequency == "주 2회" else 1,
             4 if method == "에스프레소 머신" else 3 if method == "핸드 드립" else 2 if method == "커피메이커" else 1,
             4 if coffee_type == "블랙" else 3 if coffee_type == "우유 라떼" else 2 if coffee_type == "시럽 커피" else 1,
             5 if flavor == "고소한, 구운" else 4 if flavor == "달콤, 설탕" else 3 if flavor == "초콜릿" else 2 if flavor == "과일" else 1]

        cluster_prediction = c_model.predict(np.array(x).reshape(1, -1))[0]
        x.append(cluster_prediction)

        tag = brand_names
        cafe_prediction = rf_model.predict(np.array(x).reshape(1, -1))[0]
        predicted_cafe = tag[cafe_prediction]

        recommended_beans = list(
            cosine_sim_df[predicted_cafe]
            .sort_values(ascending=False)
            .drop(tag, axis=0)
            .head(3).index
        )
        evaluate_recommendations(predicted_cafe, recommended_beans)
