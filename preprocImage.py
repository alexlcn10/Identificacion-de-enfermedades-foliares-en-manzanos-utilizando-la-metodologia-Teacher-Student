'''
Preprocesamiento de imagen
--------------------------

Image data generator
train data gen
'''

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt


class PreprocData():

    def __init__(self,batchSize,targetSize,pathData):
        self.batch_size = batchSize        
        self.target_size = targetSize
        self.path = pathData
        self._imgdataGen = []
        self._imgdataGen_aug = []
        self._dataGenerator_aug = []
        self._dataX = ()
        self._dataY = ()

    def trainTest_dataGenerator(self,rotation,rescale,shear,zoom,flip_H,valSplit,classMode,labelNames):
        self._imgdataGen = ImageDataGenerator(rotation_range=rotation,                  # Rotar imagen                            
                                            rescale = rescale,                          # pixeles de 0 a 1 (1./255)
                                            shear_range=shear,                          # distinguir inclinación
                                            zoom_range=zoom,                            # distinguir alejamiento
                                            horizontal_flip=flip_H,                     # distinguir direccionalidad
                                            validation_split=valSplit)                  # usar para validacion        
  
        dataGenerator = self._imgdataGen.flow_from_directory(self.path,                 # direccion de carpeta a generar
                                                target_size=self.target_size,   
                                                class_mode=classMode,               
                                                batch_size= self.batch_size,             
                                                subset='training',                      # train / validation
                                                classes=labelNames,                     # etiquetas
                                                seed=42, shuffle=True)

        dataGenerator_test = self._imgdataGen.flow_from_directory(self.path,            # direccion de carpeta a generar
                                                target_size=self.target_size,   
                                                class_mode=classMode,               # 
                                                batch_size= self.batch_size,             
                                                subset='validation',                      # train / validation
                                                classes=labelNames,                 # etiquetas
                                                seed=42, shuffle=True)                                                

        self._dataX, self._dataY = dataGenerator.next()
        testX, testY = dataGenerator_test.next()
        print("Forma de Datos de train")
        print(f'X: {self._dataX.shape} y: {self._dataY.shape}')
        print("Forma de Datos de validation")
        print(f'X: {testX.shape} y: {testY.shape}')
        return self._dataX, self._dataY, testX, testY  

    # Devolver forma para input de modelos de redes
    def get_dataShape(self):
        return (self._dataX.shape[1], self._dataX.shape[2], self._dataX.shape[3])
    
    def augmentation_dataGenerator(self,rotation,rescale,shear,zoom,flip_H,classMode,labelNames):
        self._imgdataGen_aug = ImageDataGenerator(rotation_range=rotation,              # Rotar imagen                            
                                                rescale = rescale,                      # pixeles de 0 a 1 (1./255)
                                                shear_range=shear,                      # distinguir inclinación
                                                zoom_range=zoom,                        # distinguir alejamiento
                                                horizontal_flip=flip_H)                 # distinguir direccionalidad
  
        self._dataGenerator_aug = self._imgdataGen_aug.flow_from_directory(self.path,             # direccion de carpeta a generar
                                                        interpolation='bilinear',
                                                        target_size=self.target_size,   
                                                        class_mode=classMode,               # 
                                                        batch_size= self.batch_size,                                                             
                                                        classes=labelNames,                 # etiquetas
                                                        seed=42, shuffle=True)
   
        dataX_aug, dataY_aug = self._dataGenerator_aug.next()        
    
        return dataX_aug,dataY_aug
    
    def getDataGen_aug(self):
        return self._dataGenerator_aug

    def GeneratorExamples_plot(self,dataX,n_rows,n_columns,title,sample):    
        fig, axes = plt.subplots(n_rows,n_columns)
        fig.suptitle(title, fontsize="x-large")

        i = 0
        for batch in self._imgdataGen_aug.flow(dataX[sample].reshape((1,self.target_size[0], self.target_size[1],3)),batch_size=1):
            axes[i//n_columns,i%n_columns].imshow(image.array_to_img(batch[0]))
            i += 1
            if i == 6:
                break    
        fig.set_size_inches(12, 8)
        plt.show()

        
    

    