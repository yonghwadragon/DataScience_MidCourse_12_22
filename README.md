# DataScience_MidCourse_12_22
데이터 분석 및 머신러닝 알고리즘 비교 실험 프로젝트

# 데이터사이언스 중급 12_22

## 📌 개요
이 프로젝트는 **실제 데이터를 활용한 분석 및 다양한 머신러닝 알고리즘 비교**를 수행한 프로젝트입니다.
EDA(탐색적 데이터 분석), 회귀 분석, 의사결정 트리, SVM, 랜덤 포레스트 등의 모델을 실험적으로 적용해 보았습니다.

## 🛠️ 사용한 기술
- `Statsmodels` 및 `Scikit-learn`을 활용한 릿지, 라소 회귀 분석
- `Matplotlib`, `Seaborn`을 활용한 데이터 시각화
- `DecisionTreeClassifier`, `SVC`, `RandomForestClassifier`를 활용한 분류 모델 비교
- Confusion Matrix, Precision, Recall, F1-score 등 성능 평가 지표 활용
- Google Colab 환경에서 실행 가능

---

## 🔹 주요 실험 내용

### 1️⃣ EDA 및 회귀 분석 (릿지 & 라소)
- 데이터 전처리 및 탐색적 데이터 분석
- OLS(최소제곱법)과 정규화된 릿지 및 라소 회귀 적용
- 릿지 회귀 계수와 라소 회귀 계수의 변화 비교

#### ✔️ 실행 코드 예시 (릿지 회귀 분석)
```python
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn import datasets

# 당뇨병 데이터셋 로드
diabetes_data = datasets.load_diabetes()
df = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
df['target'] = diabetes_data.target

# 절편(intercept) 추가
X = sm.add_constant(df.drop('target', axis=1))
y = df['target']

# 릿지 회귀 수행
ridge_regression = sm.OLS(y, X).fit_regularized(L1_wt=0, alpha=1.0)
print(ridge_regression.params)
```

#### ✔️ 결과

- 릿지 회귀 계수와 라소 회귀 계수의 차이를 비교하는 그래프 출력
- (이미지결과_1.png), (이미지결과_2.png)

---

### 2️⃣ 의사결정 트리 (Decision Tree)
- `DecisionTreeClassifier`를 활용한 모델 학습 및 시각화
- Confusion Matrix를 통해 모델 성능 평가

#### ✔️ 실행 코드 예시
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 모델 학습 및 시각화
model = DecisionTreeClassifier(max_depth=2, random_state=0)
model.fit(X, y)
plt.figure(figsize=(20,10))
plot_tree(model, filled=True)
plt.show()
```

#### ✔️ 결과
- 의사결정 트리 구조 시각화
- `이미지결과_3.png`, `이미지결과_4.png`

---

### 3️⃣ SVM (Support Vector Machine)
- SVM을 활용한 데이터 분류
- Confusion Matrix 및 성능 지표 출력

#### ✔️ 실행 코드 예시
```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# SVM 모델 생성 및 학습
model = SVC(kernel='linear', C=1.0, random_state=0)
model.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy: {accuracy:.2f}')
```

#### ✔️ 결과
- SVM의 분류 성능 시각화 및 Confusion Matrix
- `이미지결과_5.png`, `이미지결과_6.png`

---

### 4️⃣ 랜덤 포레스트 (Random Forest)
- `RandomForestClassifier`를 활용한 다중 클래스 분류
- 모델 성능 평가 (정밀도, 재현율, F1-score 출력)

#### ✔️ 실행 코드 예시
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 랜덤 포레스트 모델 생성 및 학습
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_train, y_train)

# 테스트 데이터 예측
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))
```

#### ✔️ 결과
- Random Forest 모델 성능 평가
- `이미지결과_7.png`

---

## 🔗 관련 기술 및 패키지
- `Scikit-learn`
- `Statsmodels`
- `Matplotlib`
- `Seaborn`
- `NumPy`, `Pandas`
