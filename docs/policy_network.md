# Policy Network
The policy network model allows the user to train a per-region neural network
to predict the optimal policy given the inputs (typically the size of the loop).
To ensure minimal overhead, these neural networks are kept very small with
only two hidden layers. The policy network model can be selected by setting the
environment variable `APOLLO_INIT_MODEL="PolicyNet"`.

## Hyperparameters
Hyperparameters can be set for the policy network using environment variables.
To train the network for the first 100 cycles and then only train every
10<sup>th</sup> cycle, the user could would set the following environment
variables:

`APOLLO_INIT_TRAIN=100`

`APOLLO_TRAIN_FREQ=10`

The learning rate of the optimizer can be set as follows:

`APOLLO_INIT_MODEL="PolicyNet,lr=0.01"`

The policy network uses several exponential moving averages to improve the
quality of training. The weighting factors can be set as:

`APOLLO_INIT_MODEL="PolicyNet,beta=0.5,beta1=0.5,beta2=0.9"`

where beta is used for computing the average reward and beta1 and beta2 are
used by the optimizer (see further technical notes).

The scaling used to normalize the inputs is set by (also see notes below):

`APOLLO_INIT_MODEL="PolicyNet,scale=44.36"`

Using the policy network may result in overhead that worssens performance in
cases where the average execution time of a region is too low. In this case,
the user can set a threshold where training and execution will be skipped for
such regions and will only use the default policy. For example, to ensure
that regions that take less than 0.001 seconds to execute only use the
default policy, one can set:

`APOLLO_INIT_MODEL="PolicyNet,threshold=0.001"`

Note that the values used in the examples shown above are the default values
used by Apollo if not set explicitly.

## Further Technical Notes
The policy network is a standard neural network that outputs the probabilities
of selecting policy options given the input. As training continues, the policy
network should converge to selecting the optimal policy with probability of
unity. The policy network implemented here uses the standard REINFORCE
algorithm for policy gradient based reinforcement learning.

The neural networks contain two hidden layers and a single output layer. The
hidden layers use the ReLU activation function and the output layer uses a
softmax activation. The size of the hidden layers is set as the average of
the number of inputs and number of policy options. The size of the output
layer is set to the number of policy options. The network is optimized using
the Adam optimizer.

Typically, the input of the neural network should be the size of the for loop.
Nerual networks train best when the inputs are of order unity, however the
raw input value could scale to very large orders of magnitude. To normalize
these inputs, Apollo first computes the natural log of the input to
improve dynamic range and then divides by a scaling factor so that inputs are
constrained between zero and one. By default, this scale factor is set to 
ln(2<sup>64</sup>) under the assumption of a maximum loop size constrained by
an unsigned long.

The reward value used during training is the negative of the natural log of
the loop execution time. Policy gradient methods suffer from high variance
in the estimates of the gradients. Apollo uses a standard technique for
reducing variance by subtracting a baseline value from the reward. The
baseline used by Apollo is an exponential moving average of all previous
rewards.
