from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dropout, Activation, \
    Input, Dense, Reshape, Lambda, BatchNormalization, Bidirectional
from keras.optimizers import Adam, RMSprop, Adadelta
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM
from keras import regularizers
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

import time
from parameter import *
from utils import *
import cv2

class CRNN:
    def __init__(self, img_w, img_h, num_classes, max_text_len,
         drop_rate, weight_decay, learning_rate = 1e-3, training = True):

        self.img_w = img_w
        self.img_h = img_h
        self.num_classes = num_classes
        self.max_text_len = max_text_len
        self.learning_rate = learning_rate
        self.drop_rate = drop_rate
        self.weight_decay = weight_decay

        self.model = self.build_model()


    def ctc_lambda_func(self, args):
        y_pred, labels, input_length, label_length = args
        y_pred = y_pred[:, 2:, :]
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    def build_model(self, training = True):
        input_shape = (self.img_w, self.img_h, 1)     # (128, 64, 1)

        inputs = Input(name='the_input', shape=input_shape, dtype='float32')

        x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), kernel_initializer='he_normal', name='conv1')(inputs)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)

        x = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), kernel_initializer='he_normal', name='conv2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)

        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), kernel_initializer='he_normal', name='conv3')(x)
        x = Activation('relu')(x)

        x = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), kernel_initializer='he_normal', name='conv4')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1, 2), name='pool3')(x)

        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), kernel_initializer='he_normal', name='conv5')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), kernel_initializer='he_normal', name='conv6')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1, 2), name='pool4')(x)
        x = Dropout(self.drop_rate)(x)

        x = Conv2D(512, (2, 2), padding='same', kernel_regularizer=regularizers.l2(self.weight_decay), kernel_initializer='he_normal', name='con7')(x)
        x = Activation('relu')(x)
        x = Dropout(self.drop_rate)(x)

        # CNN to RNN
        x = Reshape(target_shape=((32, 2048)), name='map-to-sequence')(x)

        lstm1_merged = Bidirectional(LSTM(256, return_sequences=True, dropout = self.drop_rate))(x)
        lstm2_merged = Bidirectional(LSTM(256, return_sequences=True, dropout = self.drop_rate))(lstm1_merged)

        x = Dense(self.num_classes, name='dense_classes')(lstm2_merged)

        y_pred = Activation('softmax', name='softmax')(x)

        labels = Input(name='the_labels', shape=[self.max_text_len], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        loss_out = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        if training:
            model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
            model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer= RMSprop(lr = self.learning_rate))
            return model
        else:
            model = Model(inputs=[inputs], outputs=y_pred)
            return model

    def train(self, train_dataloader, val_dataloader, epoch):

        save_dir = './save_model/crnn/'
        if not os.path.exists(save_dir): # if there is no exist, make the path
            os.makedirs(save_dir)
        model_path = save_dir + '{epoch:02d}-{val_loss:.4f}.hdf5'

        checkpoint = ModelCheckpoint(filepath=model_path,
                         monitor='val_loss', verbose=1, mode='min', period=1, save_best_only = True)
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 10, verbose = 1, min_lr = 1e-10)

        start_time = time.time()
        self.history = self.model.fit_generator(generator=train_dataloader.generator(),
                steps_per_epoch=int(train_dataloader.total_batch // train_dataloader.batch_size),
                epochs=epoch,
                callbacks=[checkpoint, reduce_lr],
                validation_data=val_dataloader.generator(),
                validation_steps=int(val_dataloader.total_batch // val_dataloader.batch_size))
        print("\n Training --- %s sec---" %(time.time() - start_time))
        return self.history

    def saved_model_use(self, save_dir = None):
        if save_dir == None:
            return print('No path')

        self.model = self.build_model(training=False)
        self.model.load_weights(save_dir)

        return print("Loaded model from '{}'".format(save_dir))

    def predict(self, img_path):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = img.astype(np.float32)
        img = cv2.resize(img, (self.img_w, self.img_h))
        img = (img / 255.0)
        img = img.T
        img = np.expand_dims(img, axis=-1)
        img = np.expand_dims(img, axis=0)

        net_out_value = self.model.predict(img)
        return net_out_value
