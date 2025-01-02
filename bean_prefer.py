# import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")

# 현재 디렉터리 경로
#current_dir = os.path.dirname(os.path.abspath(__file__))

# 모델 파일 경로
#model_dir = os.path.join(current_dir, 'model')

# 모델 로드
c_model = joblib.load("./model/Kmeans_model.joblib")
rf_model = joblib.load("./model/모든 데이터 학습_best_rf_model.joblib")

# 사용자 입력 함수 (올바른 입력값 받을 때까지 반복)
def get_valid_input(prompt, valid_options):
    while True:
        user_input = input(prompt)
        if user_input in valid_options:
            return valid_options[user_input]
        print("잘못된 입력입니다. 다시 시도하세요.")

# 데이터 로드
data = pd.read_csv("./model/원두와 5가지 지표.csv")
data.set_index("Bean", inplace=True)
cosine_sim = cosine_similarity(data)
cosine_sim_df = pd.DataFrame(cosine_sim, index=data.index, columns=data.index)

brand_names = ["TheVenti", "Mega", "Paik", "Starbucks", "Ediya", "Compose", "Twosome"]

# 불호 리스트 초기화
dislike_list = []

# 추천 원두 평가 함수
def evaluate_recommendations(base_bean, recommended_beans, dislike_list):
    liked_beans = []  # 호로 평가된 원두를 저장

    while True:
        print("\n추천 원두:")
        for i, bean in enumerate(recommended_beans, start=1):
            print(f"{i}. {bean}")

        # 사용자 평가 받기
        user_feedback = {}
        for bean in recommended_beans:
            if bean not in liked_beans and bean not in dislike_list:  # 이미 평가된 원두는 제외
                feedback = get_valid_input(f"{bean}에 대해 평가해주세요 ( 1. 호 / 2. 불호 ): ", {"1": 1, "2": 2})
                user_feedback[bean] = feedback

        # 불호인 원두를 dislike_list에 추가, 호는 liked_beans에 저장
        for bean, feedback in user_feedback.items():
            if feedback == 2 and bean not in dislike_list:
                dislike_list.append(bean)
            elif feedback == 1 and bean not in liked_beans:
                liked_beans.append(bean)

        # 모든 추천 원두가 호라면 종료
        if len(user_feedback) > 0 and all(feedback == 1 for feedback in user_feedback.values()):
            print("\n추천이 종료됩니다.")
            break

        # 불호와 기존 추천 원두를 제외하고 새로운 추천 원두 보충
        print("\n불호로 평가된 원두를 제외하고 새로운 추천을 제공합니다.")
        brand_names = ["TheVenti", "Mega", "Paik", "Starbucks", "Ediya", "Compose", "Twosome"]
        all_candidates = cosine_sim_df[base_bean].sort_values(ascending=False).drop(dislike_list + liked_beans + brand_names + [base_bean], axis=0)
        additional_beans = list(all_candidates.head(3 - len(liked_beans)).index)  # 필요한 개수만큼 새로운 원두 추가
        recommended_beans = liked_beans + additional_beans  # 기존 호 원두 + 새로운 원두

    return liked_beans

# 매핑 딕셔너리
sex_mapping = {"1": 1, "2": 0}
is_student_mapping = {"1": 1, "2": 0}
frequency_mapping = {"1": 9, "2": 7, "3": 5, "4": 3, "5": 1}
method_mapping = {"1": 4, "2": 3, "3": 2, "4": 1}
coffee_type_mapping = {"1": 4, "2": 3, "3": 2, "4": 1}
flavor_mapping = {"1": 5, "2": 4, "3": 3, "4": 2, "5": 1}

# 원두 구입 경험 여부 묻기
print("숫자만 입력해주세요.")
purchase_history = get_valid_input("원더룸에서 원두를 구입해 본 적이 있습니까? ( 1. 예 / 2. 아니오 ): ", {"1": 1, "2": 2})

if purchase_history == 1:
    # 원두 입력받기
    print("\n구입했던 원두를 아래 목록 중에서 입력해주세요:")
    print(", ".join(data.index))
    purchased_bean = input("원두 이름을 정확히 입력해주세요: ")

    if purchased_bean in data.index:
        # 구입한 원두 제외한 유사도 높은 원두 3개 출력
        recommended_beans = list(cosine_sim_df[purchased_bean].sort_values(ascending=False).drop([purchased_bean]+brand_names, axis=0).head(3).index)
        recommended_beans = evaluate_recommendations(purchased_bean, recommended_beans, dislike_list)
    else:
        print("입력하신 원두가 목록에 없습니다. 정확히 입력해주세요.")
else:
    # 원두 구입 경험이 없으면 기존 시나리오로 진행
    sex = get_valid_input("성별 입력 ( 1. 남 / 2. 여 ): ", sex_mapping)
    age = int(input("나이를 입력하세요(숫자만): "))
    if 20 <= age <= 23:
        age_cat = 1
    elif 24 <= age <= 27:
        age_cat = 2
    else:
        age_cat = 4

    is_student = get_valid_input("직업 입력 ( 1. 학생 / 2. 기타 ): ", is_student_mapping)

    frequency_int = get_valid_input(
        "커피를 마시는 빈도 입력 ( 1. 매일 / 2. 1주일에 5~6번 / 3. 1주일에 3~4번 / 4. 1주일에 2번 / 5. 1주일에 0~1번 ): ",
        frequency_mapping
    )

    method_int = get_valid_input(
        "커피 내리는 방법 입력 ( 1. 에스프레소 머신 / 2. 핸드 드립 / 3. 커피메이커 / 4. 콜드브루 ): ",
        method_mapping
    )

    coffee_type_int = get_valid_input(
        "커피 타입 입력 ( 1. 블랙 / 2. 우유 라떼 / 3. 시럽 커피 / 4. 설탕 커피 ): ",
        coffee_type_mapping
    )

    flavor_int = get_valid_input(
        "평소 즐기는 커피의 풍미 입력 ( 1. 고소한, 구운 / 2. 달콤, 설탕 / 3. 초콜릿 / 4. 과일 / 5. 꽃향 ): ",
        flavor_mapping
    )

    # 입력 데이터를 리스트로 구성
    x = [sex, age_cat, is_student, frequency_int, method_int, coffee_type_int, flavor_int]
    print(x)

    # 군집 모델로 군집 예측
    cluster_prediction = c_model.predict(np.array(x).reshape(1, -1))[0]
    x.append(cluster_prediction)

    # 랜덤 포레스트 모델로 카페 추천
    tag = ["TheVenti", "Mega", "Paik", "Starbucks", "Ediya", "Compose", "Twosome"]
    cafe_prediction = rf_model.predict(np.array(x).reshape(1, -1))[0]
    predicted_cafe = tag[cafe_prediction]

    # 추천된 카페에 따른 원두 추천
    drop_list = list(tag)
    recommended_beans = list(cosine_sim_df[predicted_cafe].sort_values(ascending=False).drop(drop_list, axis=0).head(3).index)
    recommended_beans = evaluate_recommendations(predicted_cafe, recommended_beans, dislike_list)
