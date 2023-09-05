import numpy as np
from keras import Input
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Add, Multiply,\
     GlobalAveragePooling1D, Concatenate, GRU, BatchNormalization, ELU, Activation, UpSampling1D, Conv1DTranspose
from keras import backend as K
from keras.models import Model

# 学习率更新以及调整

def scheduler(epoch):
    if epoch == 0:
        lr = K.get_value(model.optimizer.lr)  # keras默认0.001
        K.set_value(model.optimizer.lr, lr*100)
        print("lr changed to {}".format(lr))
    if epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        # K.set_value(model.optimizer.lr, lr * math.pow(0.99, epoch/30))
        K.set_value(model.optimizer.lr, lr / (1 + 0.0001 * epoch))
        print("lr changed to {}".format(lr))
    return K.get_value(model.optimizer.lr)

index = 213

# 数据导入
data1 = np.load('D:/blood_pressure/train1/ppg' + str(index) + '.npy', allow_pickle=True)
data2 = np.load('D:/blood_pressure/train1/abp' + str(index) + '.npy', allow_pickle=True)
data3 = np.load('D:/blood_pressure/train1/ecg' + str(index) + '.npy', allow_pickle=True)

print(data1.shape)
start = int(data1.shape[0]*0.9)
end = int(data1.shape[0])

i = 125
o = 125

def Filter(inputs, c):

    input = GlobalAveragePooling1D()(inputs)
    x = Dense(c, activation='relu')(input)
    x = Dense(c, activation='softmax')(x)

    return x

def models1(inputs):

    # 编码器
    conv1 = Conv1D(filters=128, kernel_size=7, strides=1)(inputs)
    conv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv1)
    conv1 = ELU()(conv1)

    conv2 = Conv1D(filters=128, kernel_size=7, strides=1)(conv1)
    conv2 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv2)
    conv2 = ELU()(conv2)
    conv2 = MaxPooling1D(pool_size=2, strides=2)(conv2)

    conv3 = Conv1D(filters=128, kernel_size=5, strides=1)(conv2)
    conv3 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv3)
    conv3 = ELU()(conv3)

    conv4 = Conv1D(filters=128, kernel_size=5, strides=1)(conv3)
    conv4 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv4)
    conv4 = ELU()(conv4)
    conv4 = MaxPooling1D(pool_size=2, strides=2)(conv4)

    conv5 = Conv1D(filters=128, kernel_size=3, strides=1)(conv4)
    conv5 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv5)
    conv5 = ELU()(conv5)

    conv6 = Conv1D(filters=128, kernel_size=3, strides=1)(conv5)
    conv6 = BatchNormalization(momentum=0.99, epsilon=0.001)(conv6)
    conv6 = ELU()(conv6)
    conv6 = MaxPooling1D(pool_size=2, strides=2)(conv6)

    # 通道信息筛选器
    filter1 = Filter(conv1, 128)
    filter2 = Filter(conv2, 128)
    filter3 = Filter(conv3, 128)
    filter4 = Filter(conv4, 128)
    filter5 = Filter(conv5, 128)
    filter6 = Filter(conv6, 128)

    # 时序信息提取器
    gru1 = GRU(128, activation='tanh')(conv1)
    gru2 = GRU(128, activation='tanh')(conv2)
    gru3 = GRU(128, activation='tanh')(conv3)
    gru4 = GRU(128, activation='tanh')(conv4)
    gru5 = GRU(128, activation='tanh')(conv5)
    gru6 = GRU(128, activation='tanh')(conv6)

    # 通道与时序结合
    filter_gru1 = Add()([filter1, gru1])
    filter_gru2 = Add()([filter2, gru2])
    filter_gru3 = Add()([filter3, gru3])
    filter_gru4 = Add()([filter4, gru4])
    filter_gru5 = Add()([filter5, gru5])
    filter_gru6 = Add()([filter6, gru6])

    # 解码器
    Tconv1 = Conv1DTranspose(filters=128, kernel_size=3, strides=1)(conv6)
    Tconv1 = BatchNormalization(momentum=0.99, epsilon=0.001)(Tconv1)
    Tconv1 = ELU()(Tconv1)
    Tconv1 = Multiply()([filter_gru6, Tconv1])

    Tconv2 = Conv1DTranspose(filters=128, kernel_size=3, strides=2)(Tconv1)
    Tconv2 = BatchNormalization(momentum=0.99, epsilon=0.001)(Tconv2)
    Tconv2 = ELU()(Tconv2)
    Tconv2 = Multiply()([filter_gru5, Tconv2])

    Tconv3 = Conv1DTranspose(filters=128, kernel_size=5, strides=1)(Tconv2)
    Tconv3 = BatchNormalization(momentum=0.99, epsilon=0.001)(Tconv3)
    Tconv3 = ELU()(Tconv3)
    Tconv3 = Multiply()([filter_gru4, Tconv3])

    Tconv4 = Conv1DTranspose(filters=128, kernel_size=5, strides=2)(Tconv3)
    Tconv4 = BatchNormalization(momentum=0.99, epsilon=0.001)(Tconv4)
    Tconv4 = ELU()(Tconv4)
    Tconv4 = Multiply()([filter_gru3, Tconv4])

    Tconv5 = Conv1DTranspose(filters=128, kernel_size=7, strides=1)(Tconv4)
    Tconv5 = BatchNormalization(momentum=0.99, epsilon=0.001)(Tconv5)
    Tconv5 = ELU()(Tconv5)
    Tconv5 = Multiply()([filter_gru2, Tconv5])

    Tconv6 = Conv1DTranspose(filters=128, kernel_size=7, strides=2)(Tconv5)
    Tconv6 = BatchNormalization(momentum=0.99, epsilon=0.001)(Tconv6)
    Tconv6 = ELU()(Tconv6)
    Tconv6 = Multiply()([filter_gru1, Tconv6])

    out = GlobalAveragePooling1D()(Tconv6)

    return out

def models(inputs1, inputs2):

    x = models1(inputs1)

    y = models1(inputs2)

    z = Add()([x, y])
    z = Dense(o, activation='linear')(z)
    z = Dense(o, activation='linear')(z)

    out = Model(inputs=[inputs1, inputs2], outputs=[z], name="model")

    return out

inputs1 = Input(shape=(i, 1))
inputs2 = Input(shape=(i, 1))
model = models(inputs1, inputs2)
model.summary()


model.compile(loss=['mean_absolute_error'], optimizer='Adam')

filepath = "D:/blood_pressure/8BP/view/model_heat.hdf5"  # 保存模型的路径

checkpoint = ModelCheckpoint(filepath=filepath, verbose=2,
                             monitor='val_loss', mode='min', save_best_only='True')

reduce_lr = LearningRateScheduler(scheduler)  # 学习率的改变
callback_lists = [checkpoint, reduce_lr]

train_history = model.fit(x=[data1[:start], data3[:start]],
                          y=[data2[:start]], verbose=2,
                          validation_data=([data1[start:end], data3[start:end]], [data2[start:end]]),
                          class_weight=None, callbacks=callback_lists,
                          epochs=500, batch_size=128, shuffle=False)
