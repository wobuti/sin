import numpy as np
import cupy
import chainer
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.links as L
import matplotlib
import matplotlib.pyplot as plt
import six
import time
import pandas as pd
import random
import net
import argparse
from chainer import computational_graph
import math
import sys
import chainer.functions as F
parser = argparse.ArgumentParser()
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=20, type=int,
                    help='number of epochs to learn')
parser.add_argument('--unit', '-u', default=200, type=int,
                    help='number of units')
parser.add_argument('--batchsize', '-b', type=int, default=20,
                    help='learning minibatch size')
parser.add_argument('--bproplen', '-l', type=int, default=2,
                    help='length of truncated BPTT')
parser.add_argument('--gradclip', '-c', type=int, default=5,
                    help='gradient norm threshold to clip')
parser.add_argument('--test', dest='test', action='store_true')
parser.set_defaults(test=False)

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

bprop_len = args.bproplen   # length of truncated BPTT
grad_clip = args.gradclip    # gradient norm threshold to clip


def make_mini_batch(train_data, size_of_mini_batch, length_of_sequences):
    inputs_batch = []
    outputs_batch = []
    for _ in range(size_of_mini_batch):
        index = random.randint(0, len(train_data) - length_of_sequences)
        part = train_data[index:index + length_of_sequences]
        # print('part[:, 0]', part[:, 0])
        # print('part[-1, 1]', part[-1, 1])
        inputs_batch = np.append(inputs_batch, part[:, 0])
        outputs_batch = np.append(outputs_batch, part[-1, 1])
    inputs_batch = inputs_batch.reshape(-1, length_of_sequences, 1)
    outputs_batch = outputs_batch.reshape(-1, 1)
    return inputs_batch, outputs_batch


def make_prediction_initial(train_data, index, length_of_sequences):
    inputs_prediction = []
    part = train_data[index:index + length_of_sequences, 0]
    inputs_prediction = np.append(inputs_prediction, part[:])
    inputs_prediction = inputs_prediction.reshape(-1, length_of_sequences, 1)
    return inputs_prediction

train_data_path             = "./normal.npy"
num_of_input_nodes          = 1
num_of_hidden_nodes         = 2
num_of_output_nodes         = 1
length_of_sequences         = 50
num_of_training_epochs      = 2000
length_of_initial_sequences = 50
num_of_prediction_epochs    = 200
size_of_mini_batch          = 100
learning_rate               = 0.2
print("train_data_path             = %s" % train_data_path)
print("num_of_input_nodes          = %d" % num_of_input_nodes)
print("num_of_hidden_nodes         = %d" % num_of_hidden_nodes)
print("num_of_output_nodes         = %d" % num_of_output_nodes)
print("length_of_sequences         = %d" % length_of_sequences)
print("num_of_training_epochs      = %d" % num_of_training_epochs)
print("length_of_initial_sequences = %d" % length_of_initial_sequences)
print("num_of_prediction_epochs    = %d" % num_of_prediction_epochs)
print("size_of_mini_batch          = %d" % size_of_mini_batch)
print("learning_rate               = %f" % learning_rate)
print("grad_clip                   = %f" % grad_clip)
train_data = np.load(train_data_path)

# make model
model = net.RNNLM(length_of_sequences, num_of_hidden_nodes)
if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    cupy.cuda.set_allocator(cupy.cuda.MemoryPool().malloc)
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.SGD(lr=learning_rate)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

start_at = time.time()
cur_at = start_at
accum_loss = 0
cur_log_perp = xp.zeros(())
total_loss = 0
losses =[]
begin_at = time.time()
for epoch in six.moves.range(num_of_training_epochs):
    input_data, output_data = make_mini_batch(train_data, size_of_mini_batch, length_of_sequences)
    x = chainer.Variable(xp.asarray(input_data, dtype=xp.float32))
    t = chainer.Variable(xp.asarray(output_data, dtype=xp.float32))
    loss_i = model(x, t)
    losses.append(loss_i.data)
    accum_loss += loss_i
    total_loss += loss_i.data
    cur_log_perp += loss_i.data

    # if epoch == 2:
    #     with open('graph.dot', 'w') as o:
    #         o.write(computational_graph.build_computational_graph(
    #             (model.loss,)).dump())
    #     print('generated graph', file=sys.stderr)

    if (epoch+1) % bprop_len == 0:  # Run truncated BPTT
        model.zerograds()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = 0
        total_loss = 0
        optimizer.update()

    if epoch % 100 == 0:
        duration = time.time() - begin_at
        throughput = epoch * size_of_mini_batch / duration
        print("epoch: {}/{} train loss: {} test loss: {} ({})".format(epoch, num_of_training_epochs,
                                                                 total_loss/size_of_mini_batch, loss_i.data,throughput))

    if epoch % 100 == 0:
        optimizer.lr /= 1.2

plt.plot(losses)
plt.yscale('log')
plt.show()

inputs = make_prediction_initial(train_data, 0, length_of_initial_sequences)
outputs = np.empty(0)
np.save("initial.npy", inputs[0])
model.reset_state()
for epoch in range(num_of_prediction_epochs):
    x = chainer.Variable(xp.asarray(inputs, dtype=xp.float32))
    output = model.predict(x)
    inputs = np.delete(inputs, 0)
    # print("output.data", output.data)
    if args.gpu>=0:
        inputs = np.append(inputs, xp.asnumpy(output.data))
        outputs = np.append(outputs, xp.asnumpy(output.data))
    else:
        inputs = np.append(inputs, output.data)
        outputs = np.append(outputs, output.data)
    inputs = inputs.reshape((-1, length_of_initial_sequences, 1))


np.save("output.npy", outputs)


initial = np.load("initial.npy")
output = np.load("output.npy")
train_df = pd.DataFrame(train_data[:len(initial) + len(output), 0], columns=["train"])
initial_df = pd.DataFrame(initial, columns=["initial"])
output_df = pd.DataFrame(output, columns=["output"], index=range(len(initial), len(initial) + len(output)))
merged = pd.concat([train_df, initial_df, output_df])
merged.plot(style=["-", "-", "k--"], figsize=(15, 5), grid=True)
plt.show()