#1227_1_利用增加模型複雜度或抽樣訓練，達到Overfitting，再加L1、L2或dropout

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

x_train = x_train.astype('float32')[:10000]
x_test = x_test.astype('float32')[:10000]
y_train = y_train[:10000]
y_test = y_test[:10000]

x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# y 值轉成 one-hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



# 建立簡單的線性執行的模型
model = Sequential()

# 建立卷積層，filter=32,即 output space 的深度, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

# 建立卷積層，filter=64,即 output size, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(64, (3, 3), activation='relu'))

# 建立池化層，池化大小=2x2，取最大值
model.add(MaxPooling2D(pool_size=(2, 2)))

# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
model.add(Dropout(0.25))

# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
model.add(Flatten())

# 全連接層: 128個output
model.add(Dense(128, activation='relu'))

# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.5
#model.add(Dropout(0.5))

# 使用 softmax activation function，將結果分類
model.add(Dense(num_classes, activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
#Before_history
before_history = model.fit(x_train, y_train,
                           batch_size=128,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_test, y_test))


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
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
#After_history
after_history = model.fit(x_train, y_train,
                           batch_size=128,
                           epochs=epochs,
                           verbose=1,
                           validation_data=(x_test, y_test))
                 
print('利用增加模型複雜度或抽樣訓練，達到Overfitting，再加L1、L2或dropout :')        
plt.figure(figsize=(10, 7))
plt.suptitle('overfitting then add dropout')

plt.subplot(2, 2, 1)
plt.plot(range(1, epochs + 1), before_history.history['loss'], label = 'loss')
plt.plot(range(1, epochs + 1), before_history.history['val_loss'], label = 'val_loss')
plt.xlabel('before history')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1, epochs + 1), before_history.history['acc'], label = 'acc')
plt.plot(range(1, epochs + 1), before_history.history['val_acc'], label = 'val_acc')
plt.xlabel('before history')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(1, epochs + 1), after_history.history['loss'], label = 'loss')
plt.plot(range(1, epochs + 1), after_history.history['val_loss'], label = 'val_loss')
plt.xlabel('after history')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(1, epochs + 1), after_history.history['acc'], label = 'acc')
plt.plot(range(1, epochs + 1), after_history.history['val_acc'], label = 'val_acc')
plt.xlabel('after history')
plt.legend()
plt.show()