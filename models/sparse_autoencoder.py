import tensorflow as tf

class ConvAutoencoder(tf.keras.Model):
    def __init__(self, input_shape):
        super(ConvAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=input_shape),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')
        ])
    
    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded

def build_model(input_shape, optimizer='adam'):
    model = ConvAutoencoder(input_shape)
    
    if optimizer == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-5)
    elif optimizer == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    elif optimizer == 'nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])
    return model
