#1227_1_比較1227ppt的4個Optimization
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

num_classes = 10
epochs = 6

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# y 值轉成 one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def cnn(optimizer = keras.optimizers.Adadelta()):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model.fit(x_train, y_train,
                     batch_size=128,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(x_test, y_test))

#SGD
sgd_history = cnn(optimizer = keras.optimizers.SGD())
#Momentum
momentum_history = cnn(optimizer = keras.optimizers.SGD(momentum=0.9))
#Adagrad
adagrad_history = cnn(optimizer = keras.optimizers.Adagrad())
#Adam
adam_history = cnn(optimizer = keras.optimizers.Adam())

print('比較4個Optimization:')
plt.figure(figsize=(10, 7))
plt.suptitle('Optimization')

plt.subplot(2, 2, 1)
plt.plot(range(1, epochs + 1), sgd_history.history['loss'], label = 'SGD')
plt.plot(range(1, epochs + 1), momentum_history.history['loss'], label = 'SGD + Momentum')
plt.plot(range(1, epochs + 1), adagrad_history.history['loss'], label = 'Adagrad')
plt.plot(range(1, epochs + 1), adam_history.history['loss'], label = 'Adam')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1, epochs + 1), sgd_history.history['acc'], label = 'SGD')
plt.plot(range(1, epochs + 1), momentum_history.history['acc'], label = 'SGD + Momentum')
plt.plot(range(1, epochs + 1), adagrad_history.history['acc'], label = 'Adagrad')
plt.plot(range(1, epochs + 1), adam_history.history['acc'], label = 'Adam')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(1, epochs + 1), sgd_history.history['val_loss'], label = 'SGD')
plt.plot(range(1, epochs + 1), momentum_history.history['val_loss'], label = 'SGD + Momentum')
plt.plot(range(1, epochs + 1), adagrad_history.history['val_loss'], label = 'Adagrad')
plt.plot(range(1, epochs + 1), adam_history.history['val_loss'], label = 'Adam')
plt.xlabel('epochs')
plt.ylabel('val_loss')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(1, epochs + 1), sgd_history.history['val_acc'], label = 'SGD')
plt.plot(range(1, epochs + 1), momentum_history.history['val_acc'], label = 'SGD + Momentum')
plt.plot(range(1, epochs + 1), adagrad_history.history['val_acc'], label = 'Adagrad')
plt.plot(range(1, epochs + 1), adam_history.history['val_acc'], label = 'Adam')
plt.xlabel('epochs')
plt.ylabel('val_acc')
plt.legend()
plt.show()