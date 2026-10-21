import os, re, zipfile, requests
import numpy as np
import imageio.v2 as imageio
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


url = "http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
zip_path = "coil20.zip"
folder = "coil-20-proc"

if not os.path.isfile(zip_path):
    print("Downloading COIL-20...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(r.content)

if not os.path.isdir(folder):
    print("Extracting COIL-20...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")


filelist = [f for f in os.listdir(folder) if f.endswith(".png")]
print("Number of images:", len(filelist))

X_coil, y_coil = [], []
for filename in filelist:
    im = imageio.imread(os.path.join(folder, filename))
    X_coil.append(im.flatten())
    y_coil.append(int(re.search(r"obj(\d+)", filename).group(1)))

X_coil = np.array(X_coil)
y_coil = np.array(y_coil)
print("X_coil shape:", X_coil.shape)


print("Running UMAP (this may take a few minutes)...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding_coil = reducer.fit_transform(X_coil)
print("Embedding shape:", embedding_coil.shape)

k_values = [10, 20, 40, 80, 160]
print("\n| k | Reproduced UMAP accuracy (COIL-20) |")
print("|---|---|")
for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(clf, embedding_coil, y_coil, cv=cv, scoring='accuracy')
    print(f"| {k} | {scores.mean():.3f} (± {scores.std():.3f}) |")