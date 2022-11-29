#Se importan las librerías necesarias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, SGD
from matplotlib import pyplot as plt
import numpy as np

class EDOsol(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        #El dominio de la función es de [-5,5]
        x = tf.random.uniform((batch_size,1), minval= -5 , maxval= 5)

        with tf.GradientTape() as tape:

            with tf.GradientTape() as tape2:
                tape2.watch(x)
                y_pred = self(x, training=True)
            dy = tape2.gradient(y_pred, x)
            x0 = tf.zeros((batch_size, 1))
            y0 = self(x0, training=True)
            eq = x*dy+y_pred-(x**2)*tf.cos(x) #Ecuación diferencial
            ic = y_pred
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

model = EDOsol()

model.add(Dense(10, activation='tanh' , input_shape=(1,)))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation='linear'))

model.summary()

model.compile(optimizer=RMSprop(),metrics=['loss'])

x = tf.linspace(-5,5,500)
history = model.fit(x,epochs=500,verbose=1)

x_testv = tf.linspace(-5,5,500)
a = model.predict(x_testv)
plt.plot(x_testv, a, label= "Solución de la red")
plt.plot(x_testv, ((((x**2)-2)*tf.sin(x))/x) + 2*tf.cos(x), label= "Solución analítica")
plt.grid()
plt.legend()
plt.title("Soluciones de la ecuación diferencial (Solución de la red VS Solución Analítica)")
plt.show()
exit()

model.save("ejercicio2a.h5")

modelo_cargado = tf.keras.models.load_model("ejercicio2a.h5")            
