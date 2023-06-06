#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
from keras.datasets import mnist
import matplotlib.pyplot as plt


# In[2]:


# Define the generator model
def create_generator():
    model = Sequential()
    model.add(Dense(128, input_dim=100))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(784, activation='tanh'))
    return model


# In[3]:


# Define the discriminator model
def create_discriminator():
    model = Sequential()
    model.add(Dense(128, input_dim=784))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    return model


# In[4]:


# Define the GAN model
def create_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


# In[5]:


# Load the MNIST dataset
(train_images, _), (_, _) = mnist.load_data()


# In[6]:


# Normalize and flatten the images
train_images = train_images / 127.5 - 1.
train_images = train_images.reshape(train_images.shape[0], 784)


# In[7]:


# Create the models
generator = create_generator()
discriminator = create_discriminator()
gan = create_gan(generator, discriminator)


# In[8]:


# Compile the models
generator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))


# In[9]:


# Define the number of epochs and batch size
epochs = 100
batch_size = 128


# In[10]:


# Train the GAN model
for epoch in range(epochs):
    # Train the discriminator
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_images = generator.predict(noise)
    real_images = train_images[np.random.randint(0, train_images.shape[0], batch_size)]
    discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    discriminator_loss_fake = discriminator.train_on_batch(generated_images, np.zeros((batch_size, 1)))
    discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    generator_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    # Print the progress
    print(f'Epoch {epoch+1}/{epochs} discriminator loss: {discriminator_loss:.4f}, generator loss: {generator_loss:.4f}')


# In[11]:


# Generate some images with the trained generator

noise = np.random.normal(0, 1, (10, 100))
generated_images = generator.predict(noise)
generated_images = (generated_images + 1) / 2.0
for i in range(generated_images.shape[0]):
    plt.subplot(2, 5, i+1)
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray_r')
plt.show()
