from dataclasses import dataclass, field
from tensorflow import keras
import tensorflow as tf
import numpy as np

@dataclass(frozen=False, unsafe_hash=True)
class Model:
    comp: int = field(init=True, default=int, repr=False, compare=False)
    model: keras.models.Model=field(init=False)

    def build_model(self, print: str =True) -> None:
        inputs = keras.layers.Input((2,))
        layers = keras.layers.Dense(100, activation='elu')(inputs)
        layers = keras.layers.BatchNormalization()(layers)
        layers = keras.layers.Dense(100, activation='elu')(layers)
        layers = keras.layers.BatchNormalization()(layers)
        layers = keras.layers.Dense(100, activation='elu')(layers)
        layers = keras.layers.BatchNormalization()(layers)
        layers = keras.layers.Dense(100, activation='elu')(layers)
        layers = keras.layers.BatchNormalization()(layers)
        layers = keras.layers.Dense(100, activation='elu')(layers)
        layers = keras.layers.BatchNormalization()(layers)
        layers = keras.layers.Dense(100, activation='elu')(layers)
        outputs = keras.layers.Dense(self.comp)(layers)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        self.model = model

        if print:
            model.summary()

    def training(self, train_dataset: tf.data.Dataset,
                 principalComponentsTensor: tf.Tensor,
                 lr: float = 1e-3,
                 epoch_num: int = 200) -> None:

        opt = tf.keras.optimizers.Adam(lr)
        train_dataset = train_dataset.repeat(epoch_num)
        mod_num = len(train_dataset)/epoch_num
        epoch = 0

        for step, x in enumerate(train_dataset):
            with tf.GradientTape(persistent=True) as tape:
                output = self.model(x[0])

                true_field = tf.tensordot(principalComponentsTensor, tf.transpose(x[1]), 1) + \
                             tf.reshape(x[2], [-1, tf.shape(x[2])[0]])

                recon_field = tf.tensordot(principalComponentsTensor, tf.transpose(output), 1) + \
                              tf.reshape(x[2], [-1, tf.shape(x[2])[0]])

                loss = tf.reduce_sum(tf.square(true_field - recon_field))

            grads = tape.gradient(loss, self.model.trainable_weights)
            trainables = self.model.trainable_weights

            opt.apply_gradients(zip(grads, trainables))

            if ((step+1) % mod_num)==0:
                print('loss: ', loss.numpy(), 'epoch: ', epoch)
                epoch += 1

    def predict(self, sample_points: np.ndarray) -> np.ndarray:
        return self.model(sample_points).numpy()