keras version of DeepTrade
===

add two functions to keras library, first add to $PYTHON_DIR/dist-packages/keras/losses.py
```

def risk_estimation(y_true, y_pred):
    return -100. * K.mean((y_true - 0.0002) * y_pred)

```

Second, add to $PYTHON_DIR/dist-packages/keras/activations.py

```
def relu_limited(x, alpha=0., max_value=1.):
    return K.relu(x, alpha=alpha, max_value=max_value)

```
