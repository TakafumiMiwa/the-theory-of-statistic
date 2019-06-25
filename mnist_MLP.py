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

train, test = mnist.get_mnist()

# const value
batchsize = 128
LEARNING_RATE = 0.0001
EPOCHS = 30
gpu_id = -1

# make a iterator
train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize,repeat=False, shuffle=False)

class MLP(chainer.Chain):
	def __init__(self, h, d_out):
		super(MLP, self).__init__(
			liner1 = L.Linear(None, h),
			liner2 = L.Linear(None, h),
			liner3 = L.Linear(None, d_out)
	)
																								        							
	def  __call__(self, x):														
		h = F.relu(self.liner1(x))
		h = F.relu(self.liner2(h))
		return self.liner3(h)

# make the network and loss
model = L.Classifier(MLP(100, 10))

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
trainer.extend(extensions.ProgressBar())

# run 
trainer.run()

