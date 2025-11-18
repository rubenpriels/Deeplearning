# Deep learning

## Data laden

```python
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
```

## Data opsplitsen

```python
X_train = X_train_full[:-10000]
y_train = y_train_full[:-10000]
X_val = X_train_full[-10000:]
y_val = y_train_full[-10000:]
```

## Shape bekijken

```python
print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)
```

## Model maken

```python
def get_model():
  model = keras.models.Sequential()
  model.add(keras.layers.Input(shape=(32,32,3)))
  model.add(keras.layers.Rescaling(scale=1./255))
  model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu"))
  model.add(keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu"))
  model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(128, activation="relu"))
  # examenvraag: hoe gaat de laatste laag eruit zien?
  model.add(keras.layers.Dense(10, activation="softmax"))

  return model
```

## Data laden

```python
model = get_model()
model.summary()
# Je zou dit zelf moeten kunnen opstellen. Dit is wss een examenvraag
```

## Data laden

```python
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])
# examenvraag: welke loss functie ga je gebruiken?
```

## Data laden

```python
model = get_model()
model.summary()
```

## Early stopping

```python
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
```

## Model trainen

```python
history = model.fit(X_train, y_train, batch_size=32, epochs=1, callbacks=[early_stopping_cb], validation_data=(X_val, y_val))
```

## Model evalueren

```python
model.evaluate(X_test, y_test)
```

## Predict

```python
model.predict(X_test[0:10])
```

## Get predictions

```python
def get_predictions(model, X, keepdims=False):
  predicted = model.predict(X)
  return keras.ops.argmax(predicted, axis=1, keepdims=keepdims)
```
