import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
X_pen = digits.data      # shape (1797, 64)
y_pen = digits.target    # shape (1797,)

print("X_pen shape:", X_pen.shape)   # should be (1797, 64)
print("Classes:", np.unique(y_pen))  # should be [0 1 2 3 4 5 6 7 8 9]

import umap

reducer = umap.UMAP(
    n_neighbors=15,      # paper default
    min_dist=0.1,        # paper default
    n_components=2,      # 2D embedding for kNN evaluation
    random_state=42      # for reproducibility (single-threaded)
)
embedding_pen = reducer.fit_transform(X_pen)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

k_values = [10, 20, 40, 80, 160]
results = {}

for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, embedding_pen, y_pen, cv=cv, scoring='accuracy')
    results[k] = (scores.mean(), scores.std())


print("\n| k | Reproduced UMAP accuracy |")
print("|---|---|")
for k in k_values:
    mean, std = results[k]
    print(f"| {k} | {mean:.3f} (± {std:.3f}) |")


from sklearn.datasets import fetch_openml
for name, ver in [('pendigits', 1), ('pendigits', 2), ('optdigits', 1)]:
    try:
        X, y = fetch_openml(name, version=ver, return_X_y=True, as_frame=False)
        print(name, ver, X.shape, len(set(y)))
    except Exception as e:
        print(name, ver, 'failed:', e)