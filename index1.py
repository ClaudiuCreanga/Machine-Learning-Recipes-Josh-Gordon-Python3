from sklearn import tree

# Training Data
features = [
	[140,1],
	[130,1],
	[150,0],
	[170,0]
]
labels = [0,0,1,1]

# Train Classifer
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features,labels)

# Make Predictions
print(clf.predict([[160,0]]))
# Output: 0-apple, 1-orange
