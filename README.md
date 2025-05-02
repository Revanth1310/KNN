🔹 Step 1: Load and filter the dataset
Dataset: Iris dataset, filtered to keep only 2 classes (binary classification).

Why: KNN is easier to visualize and understand for binary classes.

🔹 Step 2: Normalize Features
StandardScaler: Converts features to mean = 0 and std = 1.

Why: KNN uses distances; normalization ensures fair distance calculations.

🔹 Step 3: Split Dataset
70% for training, 30% for testing using train_test_split.

🔹 Step 4: Train KNN Classifier
KNeighborsClassifier(n_neighbors=3) uses 3 nearest neighbors to vote on the class.

Model is trained using .fit() on training data.

🔹 Step 5: Predict & Evaluate
Predictions made on test data.

Evaluation metrics:

Accuracy – percentage of correct predictions.

Confusion Matrix – shows true positives, false positives, etc.

🔹 Step 6: Varying K Value
Loops through different k values to compare accuracy.

Visualizes how accuracy changes with k.

🔹 Step 7: Visualize Decision Boundaries
Plots how the KNN model separates classes in 2D using contour plots.

Helps understand how the decision boundary shifts with data.
