# Writing Our First Classifier - Machine Learning Recipes #5 - https://youtu.be/AoeEHqVSNOw

from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self,X_train,y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        # Distance from test point to first training point
        best_dist = euc(row, self.X_train[0]) # Get the first one
        best_index = 0 #index
        for i in range(1, len(self.X_train)): # Iterate over all other training points
            dist = euc(row, self.X_train[i])
            if dist < best_dist: # Found closer, update
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

# partition into training and testing sets
from sklearn.cross_validation import train_test_split

# test_size=0.5 -> split in half
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# classifier
my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

# predictions
predictions = my_classifier.predict(X_test)
#print(predictions)

# test our classifier
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))


