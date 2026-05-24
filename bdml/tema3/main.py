from random_forest import RandomForestClassifier

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():
    iris = load_iris(as_frame=True)
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target, name="target")

    X = features.to_numpy()
    y = target.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    forest = RandomForestClassifier(
        n_estimators=25,
        max_depth=8,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        random_state=42,
    )
    forest.fit(X_train, y_train)
    print(f"Training accuracy: {forest.score(X_train, y_train):.3f}")
    print(f"Test accuracy: {forest.score(X_test, y_test):.3f}")


if __name__ == "__main__":
    main()
