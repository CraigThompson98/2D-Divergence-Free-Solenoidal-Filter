# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 09:24:57 2023

@author: CraigThompson98
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import use
use('Agg')

# Ensure you're using TensorFlow v2 and check GPU availibility and disable interactive logging
print("TensorFlow Version: ",tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

InputNodes = 2
OutputNodes = 2
NodesPerLayer = 20
HiddenLayers = 8
StepsAdam = 1000
ActivationFunciton = 'gelu'
OptimisationMethod = 'adam'

DataXY = np.loadtxt('Data/ExpPoints15.txt', delimiter=',')
DataUV = np.loadtxt('Data/Mean15.txt', delimiter=',')
xy = tf.constant(DataXY, dtype=tf.float32)

InputLayer = tf.keras.layers.Input(shape=(InputNodes,))
Layer = InputLayer
for i in range(HiddenLayers):
  Layer = tf.keras.layers.Dense(NodesPerLayer, activation=ActivationFunciton, kernel_initializer=tf.keras.initializers.GlorotUniform())(Layer)
OutputLayer = tf.keras.layers.Dense(OutputNodes)(Layer)
Model = tf.keras.models.Model(inputs=InputLayer, outputs=OutputLayer)
Model.summary()

#xy is a ndarray with shape (49053,2)
#The model produces velocity uv with shape (49053,2)
def customLoss(xy):
    def lossFunction(y_true, y_pred):
        with tf.GradientTape() as tape:
            tape.watch(xy)
            uv = Model(xy)
        grads = tape.batch_jacobian(uv, xy)
        DivergenceLoss = tf.reduce_mean(tf.square(grads[:,0,0] + grads[:,1,1]))
        DataLoss = tf.reduce_mean(tf.square(y_true - y_pred))
        TotalLoss = tf.reduce_mean(DataLoss + DivergenceLoss)
        return TotalLoss
    return lossFunction

Model.compile(optimizer=OptimisationMethod,
              loss=customLoss(xy),
              metrics=['accuracy'])

Model.fit(DataXY, DataUV, epochs=StepsAdam, verbose=2)

x = xy[:,0]

x = DataXY[:,0]
y = DataXY[:,1]
u = DataUV[:,0]
v = DataUV[:,1]
# Define the regular grid for interpolation
grid_x, grid_y = np.mgrid[min(x):max(x):500j, min(y):max(y):500j]
# Interpolate the velocity components onto the regular grid
u_interp = griddata((x, y), u, (grid_x, grid_y), method='cubic')
v_interp = griddata((x, y), v, (grid_x, grid_y), method='cubic')
# Calculate velocity magnitude
velocity_magnitude = np.sqrt(u_interp**2 + v_interp**2)
# Calculate the gradient of interpolated velocity components
dx = np.gradient(grid_x, axis=0)
du = np.gradient(u_interp, axis=0)
dy = np.gradient(grid_y, axis=1)
dv = np.gradient(v_interp, axis=1)
dudx = du/dx
dvdy = dv/dy
# Calculate divergence
divergence = dudx + dvdy
# Plotting
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
# Plot velocity magnitude
axs[0].imshow(velocity_magnitude.T, extent=(min(x), max(x), min(y), max(y)), cmap='jet', origin='lower')
axs[0].set_aspect('equal')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title('Velocity Magnitude OG')
# Plot divergence
divergence_levels = np.linspace(-10, 10, 21)
divergence_cmap = plt.cm.get_cmap('RdBu_r')
axs[1].imshow(divergence.T, extent=(min(x), max(x), min(y), max(y)), cmap=divergence_cmap, origin='lower', vmin=-10, vmax=10)
axs[1].set_aspect('equal')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_title('Divergence OG')
# Add colorbars
cbar1 = fig.colorbar(axs[0].images[0], ax=axs[0])
cbar1.set_label('Magnitude')
cbar2 = fig.colorbar(axs[1].images[0], ax=axs[1], ticks=divergence_levels, format='%.2f')
cbar2.set_label('Divergence')
plt.tight_layout()
plt.savefig('OG.png')
plt.close(fig)

UVPredicted = Model(DataXY)
u = UVPredicted[:,0]
v = UVPredicted[:,1]
#Plot
u_interp = griddata((x, y), u, (grid_x, grid_y), method='cubic')
v_interp = griddata((x, y), v, (grid_x, grid_y), method='cubic')
# Calculate velocity magnitude
velocity_magnitude = np.sqrt(u_interp**2 + v_interp**2)
# Calculate the gradient of interpolated velocity components
dx = np.gradient(grid_x, axis=0)
du = np.gradient(u_interp, axis=0)
dy = np.gradient(grid_y, axis=1)
dv = np.gradient(v_interp, axis=1)
dudx = du/dx
dvdy = dv/dy
# Calculate divergence
divergence = dudx + dvdy
# Plotting
fig, axs = plt.subplots(2, 1, figsize=(8, 8))
# Plot velocity magnitude
axs[0].imshow(velocity_magnitude.T, extent=(min(x), max(x), min(y), max(y)), cmap='jet', origin='lower')
axs[0].set_aspect('equal')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].set_title('Velocity Magnitude Pred')
# Plot divergence
divergence_levels = np.linspace(-10, 10, 21)
divergence_cmap = plt.cm.get_cmap('RdBu_r')
axs[1].imshow(divergence.T, extent=(min(x), max(x), min(y), max(y)), cmap=divergence_cmap, origin='lower', vmin=-10, vmax=10)
axs[1].set_aspect('equal')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].set_title('Divergence Pred')
# Add colorbars
cbar1 = fig.colorbar(axs[0].images[0], ax=axs[0])
cbar1.set_label('Magnitude')
cbar2 = fig.colorbar(axs[1].images[0], ax=axs[1], ticks=divergence_levels, format='%.2f')
cbar2.set_label('Divergence')
plt.tight_layout()
plt.savefig('DIVFree.png')
plt.close(fig)

np.savez_compressed('SaveData', UVPredicted = UVPredicted)
Model.save('Model.h5')
print("INFO: Prediction and model have been saved!")