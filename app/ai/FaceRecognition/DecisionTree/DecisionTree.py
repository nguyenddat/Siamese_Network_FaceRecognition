from sklearn.tree import DecisionTreeClassifier

def threshold_optimize(X, y):
    """
    Optimize the threshold for a Decision Tree classifier using the given data.
    
    Parameters:
    X (array-like): Feature data.
    y (array-like): Target labels.
    
    Returns:
    DecisionTreeClassifier: The trained Decision Tree classifier with optimized threshold.
    """
    # Initialize the Decision Tree classifier
    clf = DecisionTreeClassifier(max_depth = 1, random_state=42)
    clf.fit(X, y)
    
    return clf.tree_.threshold[0]    