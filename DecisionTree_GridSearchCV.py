from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


# Step 1: Generate a moons dataset & split it into training and test datasets.
dataset = make_moons(n_samples=10000, shuffle=True, noise=0.4, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(dataset[0], dataset[1], random_state=12)


# Step 2: Declare DecisionTree & the parameters to be considered during cross-validation procedure.
dt = DecisionTreeClassifier()
parameters = {"criterion": ['gini', 'entropy'],
              "min_samples_split": list(range(2, 12, 2)),
              "min_samples_leaf": list(range(2, 12, 2)),
              "max_leaf_nodes": list(range(10, 52, 2))
              }

# Step 3: Perform cross-validation by GridSearchCV.
cv = GridSearchCV(dt, parameters)
cv.fit(X_train, y_train)


# Step 3a: Print the best parameters.
best_params = cv.best_params_
print("Best parameters:")
for key, value in best_params.items():
    print(f"\t{key}: {value}")


# Step 4: Train the whole training set with the best parameters.
dt = DecisionTreeClassifier(criterion=best_params["criterion"],
                            min_samples_split=best_params["min_samples_split"],
                            min_samples_leaf=best_params["min_samples_leaf"],
                            max_leaf_nodes=best_params["max_leaf_nodes"])
dt.fit(X_train, y_train)


# Step 5: Assess the performance of the trained model on test set.
perf = dt.score(X_test, y_test)
print("Performance of the trained model: {:0.2%}".format(perf))


"""
Best parameters:
	criterion: gini
	max_leaf_nodes: 20
	min_samples_leaf: 8
	min_samples_split: 2
Performance of the trained model: 86.20%
"""