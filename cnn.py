'''
Arquitectura de red neuronal
'''

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout, Flatten, Dense, Activation, BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras import  backend as K
from tensorflow.keras.regularizers import l1,l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

class ConvNeuralNetwork:
    def __init__(self,input,clases):
        self.input = input        
        self.clases = clases
        self.callBack_list = []        
        self._model = Sequential()        

    def __get__(self):
        return self._model

    def CapaInicial(self,filters,kernels,activation):
        self._model.add(Conv2D(filters,kernels,
                        padding='same',activation=activation,
                        input_shape = self.input))

    def BuildCapa(self,filters,kernels,capas,activation,regularizer,valRegul):                  
        for i in range(capas):
            if regularizer=='l2':
                self._model.add(Conv2D(filters,kernels,
                            padding='same',activation=activation,
                            kernel_regularizer=l2(valRegul)))
            elif(regularizer=='l1'):
                self._model.add(Conv2D(filters,kernels,
                            padding='same',activation=activation,
                            kernel_regularizer=l1(valRegul)))
            elif(regularizer=='l1l2'):
                self._model.add(Conv2D(filters,kernels,
                            padding='same',activation=activation,
                            kernel_regularizer=l1_l2(valRegul)))
            else:
                self._model.add(Conv2D(filters,kernels,
                            padding='same',activation=activation))
            
    def AddComplemento(self,batchNorm,dropout,dropVal,
                       maxPooling,strides):        
        if batchNorm:
            self._model.add(BatchNormalization())
        
        if maxPooling:
            if strides != None:
                self._model.add(MaxPooling2D((2,2), strides=strides))
            else:
                self._model.add(MaxPooling2D((2,2)))
                    

        if dropout:
            self._model.add(Dropout(dropVal))
                    
    def BuildTopModel(self,flatten,filters,activation,regularizer,valRegul):
        if flatten:
            self._model.add(Flatten())
        
        if regularizer=='l2':
            self._model.add(Dense(filters,activation=activation,kernel_regularizer=l2(valRegul)))
        elif(regularizer=='l1'):
            self._model.add(Dense(filters,activation=activation,kernel_regularizer=l1(valRegul)))
        elif(regularizer=='l1l2'):
            self._model.add(Dense(filters,activation=activation,kernel_regularizer=l1_l2(valRegul)))
        else:
            self._model.add(Dense(filters,activation=activation))
        
    def ModelOutput(self,activation):
        self._model.add(Dense(self.clases,activation=activation))

    def Compile(self,lr,loss):
        print("[INFO]: Compilando el modelo...")
        self._model.compile(loss=loss,
                    optimizer=Adam(learning_rate=lr),
                    metrics=["accuracy"])

    def CallBacks(self,eStop,reduceLr,mCheckPoint,pathModelSaved):
        if eStop:
            early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
            self.callBack_list.append(early_stop)
        if reduceLr:
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,verbose=1)
            self.callBack_list.append(reduce_lr)
        if mCheckPoint:
            model_cp = ModelCheckpoint(pathModelSaved,
                                        monitor='val_loss',
                                        verbose=0, 
                                        save_best_only=True, 
                                        save_weights_only=False,
                                        mode='auto',
                                        save_freq='epoch')
            self.callBack_list.append(model_cp)

    def Train(self,X,Y,valSplit,epochs,batchSize):
        print("[INFO]: Entrenando la red...")
        H = self._model.fit(X, Y, validation_split=valSplit,
                            batch_size=batchSize,
                            epochs=epochs,
                            callbacks=self.callBack_list)
        return H
    
    def Save(self,pathSave):
        self._model.save(pathSave)
    
    def Show_classification_report(self, test_x, test_y, batchSize_train, label_names):
        predictions = self._model.predict(test_x, batch_size=batchSize_train)
        pred = np.argmax(predictions, axis=1)
        test_y = np.argmax(test_y, axis=1)
        print(classification_report(test_y, pred, target_names=label_names))

    ## Gans Models

    #Generador
    def CapaInicialGen(self,seed_size,activation):
        self._model.add(Dense(8*8*256,activation=activation,input_dim=seed_size))
        self._model.add(Reshape((8,8,256)))

    def BuildCapaGen(self,filters,kernels,capas,strides,activation):
        for i in range(capas):
            self._model.add(Conv2DTranspose(filters,kernels,
                                            strides=strides,
                                            padding='same',
                                            activation=activation,
                                            use_bias=False))
    
    def CapaFinalGen(self,kernelSize):
        self._model.add(Conv2D(3,kernel_size=kernelSize,padding='same'))
        self._model.add(Activation("tanh"))

    # Discriminador
    def CapaInicialDisc(self,filters,kernels,strides,activation):
        self._model.add(Conv2D(filters,kernels,strides=strides,
                        padding='same',activation=activation,
                        input_shape = self.input))

    def BuildCapaDisc(self,filters,kernels,capas,strides,activation):
        for i in range(capas):
            self._model.add(Conv2DTranspose(filters,kernels,
                                            strides=strides,
                                            padding='same',
                                            activation=activation,
                                            use_bias=False))
                        
    def CapaFinalDisc(self,dropout,dropVal):
        self._model.add(Flatten())
        if dropout:
            self._model.add(Dropout(dropVal))
        self._model.add(Dense(1, activation='sigmoid'))
    
    # Complementos de red neuronal                                    
    def AddCompGan(self,batchNorm,momentum,dropout,dropVal):        
        if batchNorm:
            self._model.add(BatchNormalization(momentum=momentum))
        if dropout:
            self._model.add(Dropout(dropVal))                                       
                            