import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import random


dataset=keras.datasets.mnist
(train_features,train_labels),(test_features,test_labels)=dataset.load_data()
scalar=MinMaxScaler()
train_features2=np.asarray(train_features).reshape(-1,28*28)
test_features2=np.asarray(test_features).reshape(-1,28*28)
test_features3,train_features3= test_features/255.0,train_features/255.0


tranformed_xtrain=scalar.fit_transform(train_features2,(0,1))
tranformed_xtest=scalar.fit_transform(test_features2,(0,1))

model=MLPClassifier(hidden_layer_sizes=128,activation="relu",batch_size=200)
model.fit(tranformed_xtrain,train_labels)





predict=model.predict(tranformed_xtest)
predict2=model.predict(tranformed_xtrain)
acc= accuracy_score(y_true=test_labels,y_pred=predict)
acctrain=accuracy_score(y_true=train_labels,y_pred=predict2)

print(acc,acctrain)


i=random.randint(0,tranformed_xtest.shape[0])
predicted_number=model.predict(tranformed_xtest[i].reshape(-1,28*28))
plt.gray()
plt.imshow(test_features[i])
plt.show()
print(f"the factual number is {test_labels[i]} and predicted number={float(predicted_number)}")


model2=keras.Sequential()
model2.add(keras.layers.Flatten(input_shape=(28,28)))
model2.add(keras.layers.Dense(units=128,activation="relu"))
model2.add(keras.layers.Dense(units=128,activation="relu"),)
model2.add(keras.layers.Dense(units=128,activation="relu"))
model2.add(keras.layers.Dense(units=10,activation="softmax"))
model2.compile(loss=tf.losses.sparse_categorical_crossentropy,optimizer=tf.optimizers.Adam(),metrics=["accuracy"])


model2.summary()
hist=model2.fit(train_features3,train_labels,epochs=1000,batch_size=256,validation_data=(test_features3,test_labels),
           callbacks=keras.callbacks.EarlyStopping(monitor="val_loss",patience=100,min_delta=5))


predict=model.predict(tranformed_xtest)
predict2=model.predict(tranformed_xtrain)
acc= accuracy_score(y_true=test_labels,y_pred=predict)
acctrain=accuracy_score(y_true=train_labels,y_pred=predict2)



i=random.randint(0,tranformed_xtest.shape[0])
predicted_number=model.predict(tranformed_xtest[i].reshape(-1,28*28))
plt.gray()
plt.imshow(test_features[i])
plt.show()

print(acc,acctrain)
predicted_deep=model2.predict(np.reshape(test_features3[i],(-1,28,28)))
for y in predicted_deep:
   for x,t in enumerate(y):
    if t==1.0:
      predicted_numberdeep= x


print(f"the factual number is {test_labels[i]} and predicted number(ANN)={float(predicted_number)}, and deep learning-based predicted number is {predicted_numberdeep}")
print(f"accuracy score of deeplearning{hist.history['val_accuracy'[-1]]}")


plt.plot(hist.history["accuracy"])
plt.show()