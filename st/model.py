from dataclasses import dataclass, field
from tensorflow import keras
import numpy as np

@dataclass(frozen=False, unsafe_hash=True)
class Model:
    comp: int = field(init=True, default=int, repr=False, compare=False)
    model: keras.models.Model=field(init=False)

    def build_model(self, print: str =True) -> keras.models.Model:
        inputs = keras.layers.Input(2)
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

    def training(self):
        pass
