# -*- coding: utf-8 -*-
# Copyright 2017 The Xiaoyu Fang. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from keras.layers import Dense, LSTM, Activation, BatchNormalization, Dropout, initializers
from renormalization import BatchRenormalization
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.models import load_model
from keras.initializers import Constant


class WindPuller(object):
    def __init__(self, input_shape, lr=0.01, n_layers=2, n_hidden=8, rate_dropout=0.2, loss='risk_estimation'):
        print("initializing..., learing rate %s, n_layers %s, n_hidden %s, dropout rate %s." %(lr, n_layers, n_hidden, rate_dropout))
        self.model = Sequential()
        self.model.add(Dropout(rate=rate_dropout, input_shape=(input_shape[0], input_shape[1])))
        for i in range(0, n_layers - 1):
            self.model.add(LSTM(n_hidden * 4, return_sequences=True, activation='tanh',
                                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal', bias_initializer='zeros',
                                dropout=rate_dropout, recurrent_dropout=rate_dropout))
        self.model.add(LSTM(n_hidden, return_sequences=False, activation='tanh',
                                recurrent_activation='hard_sigmoid', kernel_initializer='glorot_uniform',
                                recurrent_initializer='orthogonal', bias_initializer='zeros',
                                dropout=rate_dropout, recurrent_dropout=rate_dropout))
        self.model.add(Dense(1, kernel_initializer=initializers.glorot_uniform()))
        # self.model.add(BatchNormalization(axis=-1, moving_mean_initializer=Constant(value=0.5),
        #               moving_variance_initializer=Constant(value=0.25)))
        self.model.add(BatchRenormalization(axis=-1, beta_init=Constant(value=0.5)))
        self.model.add(Activation('relu_limited'))
        opt = RMSprop(lr=lr)
        self.model.compile(loss=loss,
                      optimizer=opt,
                      metrics=['accuracy'])

    def fit(self, x, y, batch_size=32, nb_epoch=100, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, **kwargs):
        self.model.fit(x, y, batch_size, nb_epoch, verbose, callbacks,
                       validation_split, validation_data, shuffle, class_weight, sample_weight,
                       initial_epoch, **kwargs)

    def save(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = load_model(path)
        return self

    def evaluate(self, x, y, batch_size=32, verbose=1,
                 sample_weight=None, **kwargs):
        return self.model.evaluate(x, y, batch_size, verbose,
                            sample_weight, **kwargs)

    def predict(self, x, batch_size=32, verbose=0):
        return self.model.predict(x, batch_size, verbose)
