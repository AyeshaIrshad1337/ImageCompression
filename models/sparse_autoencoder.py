import tensorflow as tf
class SparseAutoencoder(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, sparsity=0.05):
        super(SparseAutoencoder,self).__init__()
        self.encoder=tf.keras.layers.Dense(hidden_dim, activation='sigmoid',activity_regularizer=tf.keras.regularizers.L1(sparsity))
        self.decoder=tf.keras.layers.Dense(input_dim,activation='sigmoid')
    def call(self,inputs):
        encoded=self.encoder(inputs)
        decoded=self.decoder(encoded)
        return decoded
def build_model(input_dim, hidden_dim):
    model=SparseAutoencoder(input_dim, hidden_dim)
    model.compile(optimizer='adam',loss='mse')
    return model