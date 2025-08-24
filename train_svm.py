import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, make_moons, make_circles
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC

# Global settings
RNG_SEED = 42
OUT_DIR = "outputs"
DATA_PATH = "data/breastcancer.csv"
os.makedirs(OUT_DIR, exist_ok=True)


def load_dataset():
    """Load dataset from CSV if available, else fallback to sklearn dataset."""
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if "target" not in df.columns:
            raise ValueError("CSV must contain a 'target' column with labels.")
        X = df.drop("target", axis=1).values
        y = df["target"].values
        target_names = ["class_0", "class_1"]
        print(f"✅ Loaded dataset from {DATA_PATH}, shape={df.shape}")
    else:
        data = load_breast_cancer()
        X, y = data.data, data.target
        target_names = data.target_names
        print("✅ Loaded sklearn built-in breast cancer dataset")
    return X, y, target_names


def plot_decision_boundary(model, X, y, title, filename):
    """Plot and save decision boundary for 2D datasets."""
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.25)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors='k')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()


def main():
    # ===== Step 1: Load dataset =====
    X, y, target_names = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RNG_SEED, stratify=y
    )

    # ===== Step 2: Train SVM (Linear & RBF) =====
    pipe_linear = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", random_state=RNG_SEED))
    ])

    pipe_rbf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", random_state=RNG_SEED))
    ])

    # ===== Step 5: Cross-validation =====
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG_SEED)

    scores_linear = cross_val_score(pipe_linear, X_train, y_train, cv=cv, scoring="accuracy")
    scores_rbf = cross_val_score(pipe_rbf, X_train, y_train, cv=cv, scoring="accuracy")

    cv_df = pd.DataFrame({
        "fold": np.arange(1, len(scores_linear) + 1),
        "linear_accuracy": scores_linear,
        "rbf_accuracy": scores_rbf
    })
    cv_df.to_csv(os.path.join(OUT_DIR, "cv_scores.csv"), index=False)

    # ===== Step 4: Hyperparameter tuning =====
    param_grid = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.01, 0.1, 1]
    }
    grid = GridSearchCV(pipe_rbf, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # ===== Final test evaluation =====
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    with open(os.path.join(OUT_DIR, "classification_report_rbf.txt"), "w") as f:
        f.write("Best Params: " + str(grid.best_params_) + "\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(values_format="d")
    plt.title("Confusion Matrix — RBF SVM (test)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix_rbf.png"), dpi=150)
    plt.close()

    # ===== Step 3: Visualize 2D decision boundaries =====
    Xm, ym = make_moons(n_samples=500, noise=0.2, random_state=RNG_SEED)
    lin2d = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="linear", random_state=RNG_SEED))])
    lin2d.fit(Xm, ym)
    plot_decision_boundary(lin2d, Xm, ym, "Linear SVM — make_moons", "decision_boundary_linear_moons.png")

    rbf2d = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=10, gamma="scale", random_state=RNG_SEED))])
    rbf2d.fit(Xm, ym)
    plot_decision_boundary(rbf2d, Xm, ym, "RBF SVM — make_moons", "decision_boundary_rbf_moons.png")

    Xc, yc = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=RNG_SEED)
    lin_circ = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="linear", random_state=RNG_SEED))])
    lin_circ.fit(Xc, yc)
    plot_decision_boundary(lin_circ, Xc, yc, "Linear SVM — make_circles", "decision_boundary_linear_circles.png")

    rbf_circ = Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", C=10, gamma="scale", random_state=RNG_SEED))])
    rbf_circ.fit(Xc, yc)
    plot_decision_boundary(rbf_circ, Xc, yc, "RBF SVM — make_circles", "decision_boundary_rbf_circles.png")

    # ===== Console summary =====
    print("=== Cross-Validation Accuracy (train split) ===")
    print("Linear:", scores_linear.mean().round(4), "+/-", scores_linear.std().round(4))
    print("RBF   :", scores_rbf.mean().round(4), "+/-", scores_rbf.std().round(4))
    print("\nBest RBF Params:", grid.best_params_)
    print("\nTest set report saved to outputs/classification_report_rbf.txt")


if __name__ == "__main__":
    main()
