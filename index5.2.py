from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_distance = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)): # Iterate over all other training points
            newDistance = euc(row, self.X_train[i])
            if newDistance < best_distance:
                best_distance = newDistance
                best_index = i
        return self.y_train[best_index]


iris = datasets.load_iris()

X_train = iris.data
y_train =  iris.target

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.5)

# classifier
my_classifier = ScrappyKNN()
my_classifier.fit(X_train,y_train)

# predictions
predictions = my_classifier.predict(X_test)

print(accuracy_score(predictions, y_test))
