'''
UTILIDADES PARA CREACIÓN DE GANS
---------------------------------

preparar imagenes
funcion de perdida del discriminador
funcion de perdida del generador
calculo de variables durante los ciclos
funcion para graficar imagenes
entrenamiento de redes
generacion y guardado de imagenes individuales
'''

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow_examples.models.pix2pix import pix2pix
AUTOTUNE = tf.data.AUTOTUNE

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

from tqdm import tqdm
from PIL import Image

def PrepareImages(path_img,GENERATE_SQUARE,):
    train_preproc_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                      path_img,                                                    
                                                      image_size=(GENERATE_SQUARE, GENERATE_SQUARE),
                                                      label_mode=None,
                                                      validation_split = 0.1,
                                                      subset='training',
                                                      batch_size = 16,
                                                      shuffle = True, seed = 42)
    
    test_preproc_ds = tf.keras.preprocessing.image_dataset_from_directory(
                                                      path_img,                                                    
                                                      image_size=(GENERATE_SQUARE, GENERATE_SQUARE),
                                                      label_mode=None,
                                                      validation_split = 0.1,                                                      
                                                      batch_size = 16,
                                                      subset='validation',
                                                      shuffle = True, seed = 42)
    
    normalization_layer = layers.experimental.preprocessing.Rescaling(scale= 1./127.5, offset=-1)
    
    train_proc_ds = train_preproc_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    test_proc_ds = test_preproc_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

    train_normalized_ds = train_proc_ds.map(lambda x: normalization_layer(x))
    test_normalized_ds = test_proc_ds.map(lambda x: normalization_layer(x))
    
    return train_normalized_ds, test_normalized_ds

def Discriminator_loss(real, generated, loss_obj):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss * 0.5

def Generator_loss(generated, loss_obj):
    return loss_obj(tf.ones_like(generated), generated)

def Calc_cycle_loss(real_image, cycled_image, LAMBDA):
    loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return LAMBDA * loss1

def Identity_loss(real_image, same_image, LAMBDA):
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return LAMBDA * 0.5 * loss

def Generate_images(model, test_input):
    prediction = model(test_input)

    plt.figure(figsize=(10, 10))

    display_list = [test_input[0], prediction[0]]
    title = ['Imagen inicial', 'Imagen modificada por generador']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

@tf.function
def Train_step(real_x, real_y, generator_g, generator_f,
               discriminator_x, discriminator_y, loss_obj,
               LAMBDA, generator_g_optimizer, generator_f_optimizer,
               discriminator_x_optimizer, discriminator_y_optimizer):
    # persistent is set to True because the tape is used more than
    # once to calculate the gradients.
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        # Generator F translates Y -> X.

        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # same_x and same_y are used for identity loss.
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # calculate the loss
        gen_g_loss = Generator_loss(disc_fake_y, loss_obj)
        gen_f_loss = Generator_loss(disc_fake_x, loss_obj)

        total_cycle_loss = Calc_cycle_loss(real_x, cycled_x, LAMBDA) + Calc_cycle_loss(real_y, cycled_y, LAMBDA)

        # Total generator loss = adversarial loss + cycle loss
        total_gen_g_loss = gen_g_loss + total_cycle_loss + Identity_loss(real_y, same_y, LAMBDA)
        total_gen_f_loss = gen_f_loss + total_cycle_loss + Identity_loss(real_x, same_x, LAMBDA)

        disc_x_loss = Discriminator_loss(disc_real_x, disc_fake_x, loss_obj)
        disc_y_loss = Discriminator_loss(disc_real_y, disc_fake_y, loss_obj)

    # Calculate the gradients for generator and discriminator
    generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)

    # Apply the gradients to the optimizer
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))

def Generate_single_images(model, test_input, pathNew):
    prediction = model(test_input)
    
    for i in range(len(test_input)):
        # save a image using extension
        tf.keras.preprocessing.image.save_img(pathNew+str(i)+'c.jpg',prediction[i])

def CompararImagen(img1, img2, title1, title2):
    plt.figure(figsize=(8, 8))
    plt.subplot(121)
    plt.title(title1)
    plt.imshow(img1)

    plt.subplot(122)
    plt.title(title2)
    plt.imshow(img2)
