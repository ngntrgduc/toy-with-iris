import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

def euclidian_distances(point, data):
    return np.sqrt(np.sum((point-data) ** 2, axis=1))

class KNeighborsClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        neighbors = []
        for x_test in X_test:
            distances = euclidian_distances(x_test, self.X_train) # get distance of neighbors
            y_sorted = [y for _, y in sorted(zip(distances, self.y_train))] 
            neighbors.append(y_sorted[:self.k]) # get k-nearest neighbors
        return list(map(lambda x: max(set(x), key=x.count), neighbors))

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return sum(y_pred == y_test) / len(y_test) # accuracy score

data = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=30, shuffle=False)

accuracies = []
range_k = range(1, 11)
for k in range_k:
    classifier = KNeighborsClassifier(k=k)
    classifier.fit(X_train, y_train)
    accuracy = classifier.evaluate(X_test, y_test)
    accuracies.append(accuracy)

# Plot
plt.plot(range_k, accuracies)
plt.xticks(range_k)
plt.xlabel('k-neighbors')
plt.ylabel('Accuracy')
plt.title('Performance of KNN from scratch')
plt.show()
