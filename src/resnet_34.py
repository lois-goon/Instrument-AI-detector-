#======================================
# buiding of a RESNET 34 deep network 
# from scratch
#======================================
import numpy
import os
import tensorflow as tf
from functools import partial

# model's paremeters : 
npaDEFAULT_INPUT_SHAPE  = [224, 224,3]       
iNB_CLASSES             = 26                 # Corrigé à 26
# default definition for 2D convolutional layers : 
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size = 3, strides = 1, padding = "same", kernel_initializer = "he_normal", use_bias = False)
# definition of a residual unit : 


class ResidualUnit(tf.keras.layers.Layer):
    
    #............
    # constructor
    #............
    def __init__(self, filters, strides = 1, activation = "relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            tf.keras.layers.BatchNormalization()
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters,kernel_size = 1, strides =  strides),
                tf.keras.layers.BatchNormalization()
          ]
    #....................................................
    # get the output of the residual unit given its input 
    #....................................................
    def call( self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# making of the complete network : 
# IN : 
#      npaInputShape : shape of the input
#      iNbClasses    : number of classes to discriminate
#&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def BuildRESNET34( npdInputShape = npaDEFAULT_INPUT_SHAPE, iNbClasses = iNB_CLASSES):
    model = tf.keras.Sequential([
        # Corrigé pour utiliser l'argument 'npdInputShape'
        DefaultConv2D(64, kernel_size = 7, strides=2, input_shape = npdInputShape),
        tf.keras.layers.BatchNormalization(), 
        tf.keras.layers.Activation("relu"),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
    ])
    prev_filters = 64
    for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
        strides = 1 if filters == prev_filters else 2
        model.add(ResidualUnit(filters, strides=strides))
        prev_filters = filters
    model.add(tf.keras.layers.GlobalAvgPool2D())
    model.add(tf.keras.layers.Flatten())
    
    # --- MODIFICATION : AJOUT DU DROPOUT ---
    # On "éteint" 40% des neurones aléatoirement avant la décision finale
    model.add(tf.keras.layers.Dropout(0.4)) 
    # --- FIN DE LA MODIFICATION ---
    
    # Corrigé pour utiliser l'argument 'iNbClasses'
    model.add(tf.keras.layers.Dense(iNbClasses, activation="softmax"))
    return( model )

###################################################################################
# entry point : 
# Corrigé : commenté pour ne pas s'exécuter à l'import
# m1 = BuildRESNET34()
# print(m1.summary())
