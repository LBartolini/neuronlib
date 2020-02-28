# neuronlib

This library is a simple rapresentation of an artificial neuron written about 1 year ago (winter 2019) so everything you see and use keep in mind that this code in python refers to the python standards of that time.

Anyway the library is called `neuronlib.py` and an example is in `test.py`.

To use the neuron just instantiate an object of class `Neurone` passing to the init method the *size of the input* and the *learning rate* (default is set as 0.01).

 The activation function is the *sigmoid* but feel free to implement your own activation fucntion.

 The actual learning phase is done by calling **neuron**.*learn* where the parametres are *input data as array*, *target as integer value (0, 1)* and eventually *to_print = True* as you usually set verbose = True.

Lastly, after training, you want to be able to predict data; you do this by calling *output* passing the *input data as array*.

Enjoy!


