import tensorflow as tf
print(tf.__version__)
#print(help(tf.lite.TFLiteConverter))
tf.lite.TFLiteConverter.from_keras_model()