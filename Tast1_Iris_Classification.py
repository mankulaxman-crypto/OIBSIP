# TASK 1 - Iris Flower Classification - Laxman Manku OIBSIP
from sklearn.datasets import load_iris
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

print("Shape:", df.shape)
print(df.isnull().sum())

sns.pairplot(df, hue='species')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42, stratify=iris.target)

for name, model in [("Logistic Regression", LogisticRegression(max_iter=200)), ("Random Forest", RandomForestClassifier(random_state=42))]:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test,pred):.4f}")
    print(confusion_matrix(y_test,pred))
