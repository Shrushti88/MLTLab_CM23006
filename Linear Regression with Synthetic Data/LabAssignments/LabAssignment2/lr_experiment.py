import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

# 1. Load dataset
X, y = load_iris(return_X_y=True)

# Binary classification for simplicity
X = X[y != 2]
y = y[y != 2]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Feature scaling (VERY important for learning rate experiments)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Learning rates to test
learning_rates = [0.0001, 0.001, 0.01, 0.1]
epochs = 50

plt.figure()
print(f"Learning rate: {lr}, Final log loss: {losses[-1]:.4f}")


# 5. Train model for each learning rate
for lr in learning_rates:
    model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",
        eta0=lr,
        max_iter=1,
        warm_start=True,
        random_state=42
    )

    losses = []

    for epoch in range(epochs):
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_train)
        loss = log_loss(y_train, y_prob)
        losses.append(loss)

    plt.plot(losses, label=f"LR = {lr}")

# 6. Plot convergence
plt.xlabel("Epochs")
plt.ylabel("Log Loss")
plt.title("Effect of Learning Rate on Convergence")
plt.legend()
plt.show()
