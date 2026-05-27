# wine_classification.py

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. 데이터셋 로드
wine = load_wine()
X = wine.data
y = wine.target

print("데이터셋 이름: Wine Dataset")
print("전체 데이터 개수:", X.shape[0])
print("특성 개수:", X.shape[1])
print("클래스 이름:", wine.target_names)

# 2. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. 머신러닝 모델 생성 및 학습
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 4. 예측
y_pred = model.predict(X_test)

# 5. 정확도 출력
accuracy = accuracy_score(y_test, y_pred)

print("\n모델: Random Forest Classifier")
print("정확도:", accuracy)

print("\n분류 보고서")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

print("\n혼동 행렬")
print(confusion_matrix(y_test, y_pred))