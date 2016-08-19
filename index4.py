from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()

X = iris.data
y = iris.target

# partition into training and testing sets
from sklearn.cross_validation import train_test_split

# test_size=0.5 -> split in half
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# classifier
my_classifier = tree.DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

# predictions
predictions = my_classifier.predict(X_test)
#print(predictions)

# test our classifier
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

# Repeat using KNN
# Classifier
from sklearn.neighbors import KNeighborsClassifier

my_knn_classifier = KNeighborsClassifier()
my_knn_classifier.fit(X_train, y_train)

# predict
knn_predictions = my_knn_classifier.predict(X_test)

# test
print(accuracy_score(y_test, knn_predictions))

