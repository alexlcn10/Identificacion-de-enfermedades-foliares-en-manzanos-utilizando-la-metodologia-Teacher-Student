from tensorflow.keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import UpSampling2D, Conv2D, LeakyReLU
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
#tqdm for a progress bar when loading the dataset
from tqdm import tqdm
import os

class GeneradorHojas:
    def __init__(self,targetSize,optimizer,discriminador,generador):
        self.image_width = targetSize
        self.image_height = targetSize
        
        self.optimizer = optimizer
        self.discriminador = discriminador
        self.generador = generador

        self.channels = 3
        self.image_shape = (self.image_width,self.image_height,self.channels)

        #Amount of randomly generated numbers for the first layer of the generator.
        self.random_noise_dimension = 100

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])
        self.generator = self.build_generator()

        #A placeholder for the generator input.
        random_input = Input(shape=(self.random_noise_dimension,))

        #Generator generates images from random noise.
        generated_image = self.generator(random_input)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        #Discriminator attempts to determine if image is real or generated
        validity = self.discriminator(generated_image)

        #Combined model = generator and discriminator combined.
        #1. Takes random noise as an input.
        #2. Generates an image.
        #3. Attempts to determine if image is real or generated.
        self.combined = Model(random_input,validity)
        self.combined.compile(loss="binary_crossentropy",optimizer=optimizer)

    def get_training_data(self,datafolder):
        print("Loading training data...")

        training_data = []
        #Finds all files in datafolder
        filenames = os.listdir(datafolder)
        for filename in tqdm(filenames):
            #Combines folder name and file name.
            path = os.path.join(datafolder,filename)
            #Opens an image as an Image object.
            image = Image.open(path)
            #Resizes to a desired size.
            image = image.resize((self.image_width,self.image_height),Image.ANTIALIAS)
            #Creates an array of pixel values from the image.
            pixel_array = np.asarray(image)

            training_data.append(pixel_array)

        #training_data is converted to a numpy array
        training_data = np.reshape(training_data,(-1,self.image_width,self.image_height,self.channels))
        return training_data

    def build_generator(self):
        model = self.generador
        # Placeholder for the random noise input
        input = Input(shape=(self.random_noise_dimension,))
        #Model output
        generated_image = model(input)
        #Change the model type from Sequential to Model (functional API) More at: https://keras.io/models/model/.
        return Model(input,generated_image)

    def build_discriminator(self):
        model = self.discriminador
        input_image = Input(shape=self.image_shape)
        #Model output given an image.
        validity = model(input_image)
        return Model(input_image, validity)

    def train(self, datafolder ,epochs,batch_size,save_images_interval):
        #Get the real images
        training_data = self.get_training_data(datafolder)

        #Map all values to a range between -1 and 1.
        training_data = training_data / 127.5 - 1.

        #Two arrays of labels. Labels for real images: [1,1,1 ... 1,1,1], labels for generated images: [0,0,0 ... 0,0,0]
        labels_for_real_images = np.ones((batch_size,1))
        labels_for_generated_images = np.zeros((batch_size,1))
        H_Dloss = []
        H_Gloss = []

        for epoch in range(epochs):
            # Select a random half of images
            indices = np.random.randint(0,training_data.shape[0],batch_size)
            real_images = training_data[indices]

            #Generate random noise for a whole batch.
            random_noise = np.random.normal(0,1,(batch_size,self.random_noise_dimension))
            #Generate a batch of new images.
            generated_images = self.generator.predict(random_noise)

            #Train the discriminator on real images.
            discriminator_loss_real = self.discriminator.train_on_batch(real_images,labels_for_real_images)
            #Train the discriminator on generated images.
            discriminator_loss_generated = self.discriminator.train_on_batch(generated_images,labels_for_generated_images)
            #Calculate the average discriminator loss.
            discriminator_loss = 0.5 * np.add(discriminator_loss_real,discriminator_loss_generated)

            #Train the generator using the combined model. Generator tries to trick discriminator into mistaking generated images as real.
            generator_loss = self.combined.train_on_batch(random_noise,labels_for_real_images)
            print ("[%d/%d] Discriminator_loss: %.3f, acc: %.2f - Generator_loss: %.3f" % (epoch,epochs, discriminator_loss[0], 100*discriminator_loss[1], generator_loss))

            if epoch % save_images_interval == 0:
                self.save_images(epoch)

            H_Dloss.append(discriminator_loss)
            H_Gloss.append(generator_loss)

        #Save the model for a later use
        self.generator.save("models/GanModels/models/generadorHojas.h5")
        return H_Dloss, H_Gloss


    def save_images(self,epoch):
        #Save 25 generated images for demonstration purposes using matplotlib.pyplot.
        rows, columns = 3, 5
        noise = np.random.normal(0, 1, (rows * columns, self.random_noise_dimension))
        generated_images = self.generator.predict(noise)

        generated_images = 0.5 * generated_images + 0.5

        figure, axis = plt.subplots(rows, columns)
        plt.figure(figsize=(9,6))
        image_count = 0
        for row in range(rows):
            for column in range(columns):
                axis[row,column].imshow(generated_images[image_count, :], cmap='spring')
                axis[row,column].axis('off')
                image_count += 1
        figure.savefig("models/GanModels/img/generated_images/generated_%d.png" % epoch)
        plt.close()

    def generate_single_image(self,model_path,image_save_path):
        noise = np.random.normal(0,1,(1,self.random_noise_dimension))
        model = load_model(model_path)
        generated_image = model.predict(noise)
        #Normalized (-1 to 1) pixel values to the real (0 to 256) pixel values.
        generated_image = (generated_image+1)*127.5
        print(generated_image)
        #Drop the batch dimension. From (1,w,h,c) to (w,h,c)
        generated_image = np.reshape(generated_image,self.image_shape)

        image = Image.fromarray(generated_image,"RGB")
        image.save(image_save_path)