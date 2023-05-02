from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=30, shuffle=False)
classifier = SVC()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(f'Accuracy of Support Vector Machine: {accuracy_score(y_test, y_pred)}')