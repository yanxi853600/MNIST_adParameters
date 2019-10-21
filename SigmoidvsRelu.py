#1227_1_Sigmoid v.s. ReLu

import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# 定義分類數量
num_classes = 10
# 定義訓練週期
epochs = 6

#定義圖像寬、高
img_rows, img_cols = 28, 28

# 載入 MNIST 訓練資料
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# channels_first: 色彩通道(R/G/B)資料(深度)放在第2維度，第3、4維度放置寬與高
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)

else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
# 轉換色彩 0~255 資料為 0~1
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

#Relu
def cnn(activation = 'relu'):
    # 建立簡單的線性執行的模型
    model = Sequential()
    
    # 建立卷積層，filter=32,即 output space 的深度, Kernal Size: 3x3, activation function 採用 relu
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation=activation,
                     input_shape=input_shape))
    
    # 建立卷積層，filter=64,即 output size, Kernal Size: 3x3, activation function 採用 relu
    model.add(Conv2D(64, (3, 3), activation=activation))
    
    # 建立池化層，池化大小=2x2，取最大值
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.25
    model.add(Dropout(0.25))
    
    # Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
    model.add(Flatten())
    
    # 全連接層: 128個output
    model.add(Dense(128, activation=activation))
    
    # Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.5
    model.add(Dropout(0.5))
    
    # 使用 softmax activation function，將結果分類
    model.add(Dense(num_classes, activation='softmax'))
    
    # 編譯: 選擇損失函數、優化方法及成效衡量方式
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    # 進行訓練
    return model.fit(x_train, y_train,
                     batch_size=128,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(x_test, y_test))

sigmoid_history = cnn(activation = 'sigmoid')
relu_history = cnn(activation = 'relu')

print('Sigmoid v.s. ReLu: ')
plt.figure(figsize=(10, 7))
plt.suptitle('sigmoid vs relu')

plt.subplot(2, 2, 1)
plt.plot(range(1, epochs + 1), relu_history.history['loss'], label = 'relu')
plt.plot(range(1, epochs + 1), sigmoid_history.history['loss'], label = 'sigmoid')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(1, epochs + 1), relu_history.history['acc'], label = 'relu')
plt.plot(range(1, epochs + 1), sigmoid_history.history['acc'], label = 'sigmoid')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(1, epochs + 1), relu_history.history['val_loss'], label = 'relu')
plt.plot(range(1, epochs + 1), sigmoid_history.history['val_loss'], label = 'sigmoid')
plt.xlabel('epochs')
plt.ylabel('val_loss')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(1, epochs + 1), relu_history.history['val_acc'], label = 'relu')
plt.plot(range(1, epochs + 1), sigmoid_history.history['val_acc'], label = 'sigmoid')
plt.xlabel('epochs')
plt.ylabel('val_acc')
plt.legend()
plt.show()