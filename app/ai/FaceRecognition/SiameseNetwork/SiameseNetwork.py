import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

@register_keras_serializable()
def euclidean_distance(vectors):
    featA, featB = vectors
    sum_squared = tf.reduce_sum(tf.square(featA - featB), axis=1, keepdims=True)
    return tf.sqrt(tf.maximum(sum_squared, tf.keras.backend.epsilon()))

@register_keras_serializable()
def contrastive_loss(y_true, y_pred, margin=2.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    squared_pred = tf.square(y_pred)
    squared_margin = tf.square(tf.maximum(margin - y_pred, 0))
    loss = y_true * squared_pred + (1 - y_true) * squared_margin
    return tf.reduce_mean(loss)

class SiameseNetwork:
    def __init__(self):
        self.input_shape = (512,)
        self._build_model()
    
    def fit(self, x_train_a, x_train_b, y_train):
        history = self.model.fit(
            x = [x_train_a, x_train_b],
            y = y_train,
            batch_size = 8,
            epochs = 100,
            verbose = 1,
        )
        return history
    
    def _build_model(self):
        # Define inputs
        input_a = tf.keras.Input(shape=self.input_shape)
        input_b = tf.keras.Input(shape=self.input_shape)

        # Define a shared network (shared weights)
        shared_dense_layer = tf.keras.layers.Dense(units=512, activation="sigmoid")

        # Get embeddings from the shared network
        represented_a = shared_dense_layer(input_a)
        represented_b = shared_dense_layer(input_b)

        distance = tf.keras.layers.Lambda(
            euclidean_distance,
            output_shape = (1,)
        )([represented_a, represented_b])
        
        self.model = tf.keras.Model(inputs=[input_a, input_b], outputs=distance)
        self.model.compile(
            loss=contrastive_loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005)
        )
        self.model.summary()
        return self