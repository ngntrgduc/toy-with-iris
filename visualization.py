import iris
import matplotlib.pyplot as plt
import numpy as np

data = iris.load_data()
df = iris.load_df()
X_train, X_test, y_train, y_test = iris.load_split(test_size=30)

def overall():
    plt.plot(df)
    plt.title('Overall')
    plt.legend(data.feature_names)
    plt.show()

def lines():
    plt.yticks(np.arange(3), data.target_names.tolist())
    plt.plot(X_train, y_train, alpha=0.8)
    plt.show()

def scatter():
    for i in range(4):
        plt.scatter(X_train[:, i], y_train)
        plt.yticks(np.arange(3), data.target_names.tolist())
    plt.show()

def scatter_subplots():
    _, ax = plt.subplots()
    for i in range(4):
        ax = plt.subplot(2, 2, i+1)
        ax.scatter(X_train[:, i], y_train)
        plt.yticks(np.arange(3), data.target_names.tolist())
        plt.xlabel(data.feature_names[i])
    plt.show()

# overall()
# lines()
# scatter()
# scatter_subplots()
