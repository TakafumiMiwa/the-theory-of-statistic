import numpy as np
import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Function, FunctionNode, gradient_check, report, training, utils, Variable
from chainer import datasets, initializers, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer import training, reporter
from chainer.training import extensions
from chainer.datasets import mnist
from chainer import Chain

train, test = mnist.get_mnist(ndim=3)

# const value
batchsize = 128
LEARNING_RATE = 0.0001
EPOCHS = 30
gpu_id = 0

# make a iterator
train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize,repeat=False, shuffle=False)

class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__()
        with self.init_scope():
            self.liner1 = L.Linear(784, 500)
            self.liner2 = L.Linear(500, 200)
            self.liner3 = L.Linear(200, 10)
            
    def  __call__(self, x):														
        h = F.relu(self.liner1(x))
        h = F.relu(self.liner2(h))
        return self.liner3(h)
    

class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.cn1 = L.Convolution2D(1, 20, 5)
            self.cn2 = L.Convolution2D(20, 50, 5)
            self.fc1 = L.Linear(800, 500)
            self.fc2 = L.Linear(500, 10)
            
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.cn1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.cn2(h1)), 2)
        h3 = F.dropout(F.relu(self.fc1(h2)))
        return self.fc2(h3)
    
# make the network and loss
model = L.Classifier(MLP())
#model = L.Classifier(CNN())

# create an optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# make updater
updater = training.updaters.StandardUpdater(train_iter, optimizer, device=gpu_id)

# make a trainer
trainer = training.Trainer(updater, (EPOCHS, 'epoch'), out='result')
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))
trainer.extend(extensions.LogReport())
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']))

# Print a progress bar to stdout
#trainer.extend(extensions.ProgressBar())

# run 
trainer.run()

