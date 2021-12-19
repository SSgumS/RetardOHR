# RetardOHR
An online handwriting recognizer by words that is RETARD! Do not expect too much from it.

This model is just a sample to show how to use [POH-Db](https://github.com/SLTLabAUT/POH-Db).
It uses only WordGroup handwriting samples.

The attention decoder used in this model was based on the one referenced below that I partially upgraded to TensorFlow 2.

The model uses only X, Y, and Time for feature extraction. Some parts of the code used for feature extraction are from the model referenced below.

The architecture used in this model is as the following picture:
![model architecture](https://github.com/SSgumS/RetardOHR/blob/main/images/architecture.png)

# Credits
- [keras-monotonic-attention](https://github.com/asmekal/keras-monotonic-attention)
- [Online-Handwriting-Recognition-using-Encoder-Decoder-model](https://github.com/AbeerEisa/Online-Handwriting-Recognition-using-Encoder-Decoder-model)
- [POH-Db](https://github.com/SLTLabAUT/POH-Db)
