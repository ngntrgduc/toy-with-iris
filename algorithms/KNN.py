import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=30, shuffle=False)
accuracies = []
range_k = range(1, 11)
for k in range_k:
    neighbor = k
    classifier = KNeighborsClassifier(n_neighbors=neighbor)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    print(f'Accuracy of {neighbor}-Nearest Neighbors: {accuracy_score(y_test, y_pred)}')

plt.plot(range_k, accuracies)
plt.xticks(range_k)
plt.xlabel('k-neighbors')
plt.ylabel('Accuracy')
plt.title('Performance of KNN')
plt.show()