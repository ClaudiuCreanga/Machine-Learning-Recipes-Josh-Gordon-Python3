import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_indexes = [0,50,100]

#training data
train_target = np.delete(iris.target, test_indexes);
train_data = np.delete(iris.data,test_indexes, axis=0);

#testing data
test_target = iris.target[test_indexes]
test_data = iris.data[test_indexes]

#classifier
clf = tree.DecisionTreeClassifier();
clf.fit(train_data, train_target)

print(clf.predict(test_data))

# Visualize
# from scikit decision tree tutorial: http://scikit-learn.org/stable/modules/tree.html
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")

# show data

# print iris.feature_names  # metadata: names of the features
# print iris.target_names  # metadata: names of the different types of flowers
# print iris.data[0]  # first flower
# print iris.target[0]  # contains the labels
#
# print entire dataset
# for i in range(len(iris.target)):
# 	print("example %d: label %s, features %s" % (i, iris.target[i], iris.data[i]))

# run in browser to inspect iris dataset

# from flask import Flask
# app = Flask(__name__)
#
# @app.route("/")
# def hello():
#     return jsonify(test_data.tolist())
#
# if __name__ == "__main__":
#     app.run(debug = True)