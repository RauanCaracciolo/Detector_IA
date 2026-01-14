from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

X = np.load('X_svm.npy')
y = np.load("y_labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = LinearSVC()
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))