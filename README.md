# CNN-pytorch
Simple CNN Model with pytorch and Cifar10 dataset is used. This code was for my pytorch practice so more optimization is needed for actual usage.


## Usage
To train the model without Batch Normalization:

    $ python cnn.py

or with Batch Normalization:

    $ python cnn_bn.py


## Performance
3 epochs * 1875 steps iteration result

    cnn.py    --> training loss: 0.076 / test accuracy: 97.32
    cnn_bn.py --> training loss: 0.110 / test accuracy: 95.12


## Author
Sooyoung Moon / [@symoon94](https://twitter.com/?lang=ko)