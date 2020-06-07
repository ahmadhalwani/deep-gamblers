import tensorflow as tf
import tensorflow.keras.backend as K


class GamblerLoss(tf.keras.losses.Loss):
    """
    Gambler's loss function.

    Parameters
    ----------
    - payoff : float, default 10.0
        Payoff for a correct bet.

    Usage with compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('adam', loss=GamblerLoss(10.0))
    ```
    """

    def __init__(self, payoff=10.0, **kwargs):
        super().__init__(**kwargs)
        self.payoff = payoff

    def call(self, y_true, y_pred):
        outputs, reservation = y_pred[..., :-1], y_pred[..., -1]
        outputs = tf.clip_by_value(outputs, K.epsilon(), 1.0 - K.epsilon())
        gain = tf.einsum("ij, ij -> i", y_true, outputs)
        doubling_rate = -tf.math.log(gain + reservation / self.payoff)
        return doubling_rate

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "reward": self.reward}


def coverage(y_true, y_pred):
    """
    Computes coverage rate for `y_pred`.

    Usage with compile API:

    ```python
    model = keras.Model(inputs, outputs)
    model.compile('adam', metrics=[coverage])
    ```
    """
    m = tf.shape(y_pred)[1] - tf.constant(1, dtype=tf.int32)
    n_samples = tf.cast(tf.shape(y_pred)[0], tf.float32)
    n_abstain = tf.reduce_sum(
        tf.where(tf.argmax(y_pred, axis=1, output_type=tf.int32) == m, 1.0, 0.0)
    )
    return tf.constant(1.0) - n_abstain / n_samples
