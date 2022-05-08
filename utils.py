'''
UTILIDADES
-------------------

crear entorno inicial
modificar data frame
obtener etiquetas
crear entorno de train
'''

import sys
import os
from os import listdir
import matplotlib
import numpy as np
import pandas as pd
import csv
import cv2
import re
import matplotlib.pyplot as plt
import shutil

from sklearn.metrics import classification_report

def CreateEnvironment(pathBase):

    #Renombrar carpeta de train
    try:
        print("renombrando carpeta de train ...")
        os.rename(pathBase+'images', pathBase+'train')
    except:
        pass
    
    #Crear carpetas necesarias
    try:
        print("creando carpeta de noEtiquetadas ...")
        os.mkdir(pathBase+"noEtiquetadas")
    except:
        pass

    #Mover datos sin etiquetas a test
    if os.path.isfile(pathBase+"train/Test_0.jpg"):
        try:
            print("moviendo archivos no etiquetados...")
            for i in range(1821):    
                shutil.move(pathBase+"train/Test_"+str(i)+".jpg", "noEtiquetadas")
        except:
            pass
    else:
        print("los archivos ya han sido movidos")

    try:
        print("Creando carpetas para guardar modelos ...")
        os.mkdir("models")
        os.mkdir("models/teacherModels")
        os.mkdir("models/teacherModels/models")
        os.mkdir("models/teacherModels/img")
        os.mkdir("models/GanModels")
        os.mkdir("models/GanModels/models")
        os.mkdir("models/GanModels/img")
        os.mkdir("models/studentModels")
        os.mkdir("models/studentModels/models")
        os.mkdir("models/studentModels/img")
    except:
        pass

# Modificar DataFrame
def ModDataFrame(dataFrame):
    dataFrame.loc[dataFrame.healthy==1,'label']='saludable'
    dataFrame.loc[dataFrame.multiple_diseases==1,'label']='multiples_enfermedades'
    dataFrame.loc[dataFrame.rust==1,'label']='oxido'
    dataFrame.loc[dataFrame.scab==1,'label']='costra'    
    dataFrame = dataFrame.drop(['healthy','multiple_diseases','rust','scab'], axis=1)
    return dataFrame

def ObtenerEtiquetas(dataFrame):
    # Obtener etiquetas
    classNames = []
    for row in dataFrame['label']:
        classNames.append(row)

    etiquetas = np.unique(classNames)
    etiquetas = etiquetas.tolist()
    print("\nEtiquetas de train")
    print(etiquetas)
    return etiquetas

# Crear carpetas para cada etiqueta y renombrar archivos
def SortEnvironment(dataFrame,etiquetas,pathInit,pathEnd):
    list= dataFrame
    list ["FILE_ID_JPG"] = ".jpg"                            
    list ["FILE_ID1"] = list ["image_id"] + list ["FILE_ID_JPG"]   

    for i in etiquetas:
        try:
            print("creando sub carpetas ...")
            os.mkdir(pathEnd+str(i))           
        except:
            pass
    # mover a carpeta segun etiqueta
    for i in etiquetas:
        listnew=list[list["label"]==i]
        l=listnew["FILE_ID1"].tolist()
        j=str(i)    
        for each in l:
            try:
                shutil.copy(pathInit+each,pathEnd+j)
            except:
                pass

# Contar imagenes y mostramos la primera por carpeta y buscar tamaño minimo
def InspecTrain_img(etiquetas,pathTrain):  
    numImages = 0
    imgIndex = 1
    imgSize = []
    n_files = []

    fig = plt.figure(figsize=(10,7))

    for carpetaEtq in etiquetas:
        select_image = True    
        onlyfiles = next(os.walk(pathTrain+carpetaEtq))[2]    # cantidad de archivos por carpeta
        n_files.append(len(onlyfiles))
        for image in listdir(pathTrain+carpetaEtq):
            img = cv2.imread(pathTrain+carpetaEtq+'/'+image)
            imgSize.append([img.shape[0], img.shape[1]])        
            if select_image:
                fig.add_subplot(2, 2, imgIndex)            
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(carpetaEtq)
                select_image = False
                imgIndex += 1            
            numImages += 1
    plt.show()
    
    print(f'Numero de imagenes: {numImages}')
    print(f'Tamaño mínimo: {np.min(imgSize, axis=0)}')
    print('\nArchivos por carpeta:\n')
    for i in range(len(etiquetas)):
        print(f'{etiquetas[i]}: {n_files[i]}')

    return numImages, n_files         

# ------------ Utils para arquitecturas de red neuronal

def Plot_training_performance(H, n_epochs,path):
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']

    loss = H.history['loss']
    val_loss = H.history['val_loss']
    
    plt.style.use("ggplot")
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14, 6))
    
    ax1.plot(acc, label='Training Accuracy')
    ax1.plot(val_acc, label='Validation Accuracy')
    ax1.legend(loc='best')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0,1.05])
    ax1.set_title('Training and Validation Accuracy')

    ax2.plot(loss, label='Training Loss')
    ax2.plot(val_loss, label='Validation Loss')
    ax2.legend(loc='best')
    ax2.set_ylabel('Cross Entropy')
    ax2.set_ylim([0,10])
    ax2.set_title('Training and Validation Loss')
    ax2.set_xlabel('epoch')
    plt.show()
    
    fig.savefig(path,bbox_inches='tight')
                            

