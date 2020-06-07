# Deep Gamblers

Tensorflow implementation of [Deep Gamblers](https://arxiv.org/abs/1907.00208).

## Installation

```sh
$ pip install git+https://github.com/simaki/deep-gamblers
```

## How to use

Example: MNIST classification with abstention

```python
import tensorflow as tf
from deep_gamblers import coverage, GamblerLoss

x_tr, y_tr = ...  # Fetch MNIST

model = tf.models.Sequential([
    Conv2D(10, 4, activation="relu"),
    Conv2D(10, 4, activation="relu"),
    Conv2D(10, 4, activation="relu"),
    Conv2D(10, 4, activation="relu"),
    Flatten(),
    Dense(10 + 1, activation="relu"),
])

model.compile(optimizer="adam", loss=GamblerLoss(6.0), metrics=[coverage, "accuracy"])
model.fit(x_tr, y_tr, epochs=10)
```
