import tensorflow as tf
import numpy as np
from FactorizationMachine import FMClassifier

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler



# Prepare data:
features, target = load_wine(return_X_y=True)


# Make a train/test split using 30% test size
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                    test_size=0.2,
                                                    random_state = RANDOM_STATE)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)




lb = LabelBinarizer()
y_train_enc = lb.fit_transform(y_train)
y_test_enc = lb.transform(y_test)

model = FMClassifier(inp_dim = 13, n_classes = 3)
model.fit(X_train_scaled, y_train_enc, batch_size = 8, num_epoch = 750)

model.evaluate(X_test_scaled, y_test_enc)