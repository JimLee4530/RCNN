from read_data import *
file = 'data/train_32x32.mat'
X_raw, y_raw = getData(filename=file)
n_train = X_raw.shape[0]
y_raw[y_raw==10] = 0

n_class = 10
y_ = np.zeros((n_train,n_class))
for i in range(n_train):
    y_[i,y_raw[i]] = 1
print(y_[:10])

from keras.models import Model
from keras.layers import *
from keras import optimizers

def RCL_block(filedepth, input):
    conv1 = Conv2D(filters=filedepth, kernel_size=[3, 3], strides=(1, 1), padding='same',activation='relu')(input)
    stack2 = BatchNormalization()(conv1)

    RCL = Conv2D(filters=filedepth, kernel_size=[3, 3], strides=(1, 1), padding='same', activation='relu')

    conv2 = RCL(stack2)
    stack3 = Add()([conv1, conv2])
    stack4 = BatchNormalization()(stack3)

    conv3 = Conv2D(filters=filedepth, kernel_size=[3, 3], strides=(1, 1), padding='same',activation='relu', weights=RCL.get_weights())(stack4)
    stack5 =  Add()([conv1, conv3])
    stack6 = BatchNormalization()(stack5)

    conv4 = Conv2D(filters=filedepth, kernel_size=[3, 3], strides=(1, 1), padding='same',activation='relu', weights=RCL.get_weights())(stack6)
    stack7 =  Add()([conv1, conv4])
    stack8 = BatchNormalization()(stack7)

    return stack8

input_img = Input(shape=(32, 32, 3))
conv1 = Conv2D(filters=192, kernel_size=[5, 5], strides=(1, 1), padding='same',activation='relu')(input_img)

rconv1 = RCL_block(192, conv1)
dropout1 = Dropout(0.2)(rconv1)
rconv2 = RCL_block(192, dropout1)
maxpooling_1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(rconv2)
dropout2 = Dropout(0.2)(maxpooling_1)
rconv3 = RCL_block(192, dropout2)
dropout3 = Dropout(0.2)(rconv3)
rconv4 = RCL_block(192, dropout3)

out = MaxPool2D((16, 16), strides=(16, 16), padding='same')(rconv4)
flatten = Flatten()(out)
prediction = Dense(10, activation='softmax')(flatten)

model = Model(inputs=input_img, outputs=prediction)
adam = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=adam, loss='categorical_crossentropy',metrics=['accuracy'])

from keras.models import load_model
# model = load_model('model-RCNN.h5')
model.fit(X_raw, y_ , batch_size=64,epochs=5, verbose=1, shuffle=True)

model.save('model-RCNN.h5')

