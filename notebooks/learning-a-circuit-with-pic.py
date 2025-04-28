# %% [markdown]
# # Probabilistic Integral Circuits

# %% [markdown]
# In this notebook, we will show an alternative way of learning the parameters of tensorized folded (probabilistic) circuits.
# 
# This technique is actually based on another model class called [*Probabilistic Integral Circuit* (PIC)](https://arxiv.org/abs/2406.06494), which extends Probabilistic Circuits (PCs) by adding *integral units*, which allow modelling continuous latent variables.
# 
# Fortunately enough, we do **not** need to fully understand PICs to apply them! In fact, from an application point of view, all we need to do is replacing every folded tensor parameter with a neural net whose output is an equally-sized tensor! Therefore, the actual parameters we are going to optimize are those of such neural nets, and not the original tensors. This is it -- nothing less, nothing more.
# 
# To showcase this alternative parameter learning scheme, we will first instantiate a folded circuit as shown in the [learning a probabilistic circuit](../learning-a-circuit) notebook.

# %%
from cirkit.templates import utils, data_modalities
from cirkit.pipeline import compile
import random
import numpy as np
import torch

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Set some seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the torch device to use
device = torch.device('cuda:0')

symbolic_circuit = data_modalities.image_data(
    (1, 28, 28),                # The shape of MNIST image, i.e., (num_channels, image_height, image_width)
    region_graph='quad-graph',  # Select the structure of the circuit to follow the QuadGraph region graph
    input_layer='categorical',  # Use Categorical distributions for the pixel values (0-255) as input layers
    num_input_units=64,         # Each input layer consists of 64 Categorical input units
    sum_product_layer='cp',     # Use CP sum-product layers, i.e., alternate dense sum layers and hadamard product layers
    num_sum_units=64,           # Each dense sum layer consists of 64 sum units
    sum_weight_param=utils.Parameterization(
        activation='none',      # Do not use any parameterization
        initialization='normal' # Initialize the sum weights by sampling from a standard normal distribution
    )
)
circuit = compile(symbolic_circuit)

# %% [markdown]
# The one above is the very same circuit from the [learning a circuit](../learning-a-circuit) notebook. Let's now print some stuff related to its first and second layer.

# %%
print(circuit.nodes[0])
print(circuit.nodes[0].probs().shape)

# %% [markdown]
# The first layer is an input categorical layer, which models the 784 gray-scale pixels using a folded tensor of shape (784, 64, 1, 256).

# %%
print(circuit.nodes[1])
print(circuit.nodes[1].weight().shape)

# %% [markdown]
# The second layer is a TorchDenseLayer whose shape is (1568, 64, 64).

# %% [markdown]
# ## Converting PCs to Quadrature PCs (QPCs)

# %% [markdown]
# As we mentioned earlier, all we need to do is replacing every folded tensor parameter with a neural net whose output returns an equally-sized tensor. Therefore, instead of training tensors, we will be training neural nets that output tensors.
# 
# We can do such *replacement* in one line by using the API ```pc2qpc```. Here QPC stands for *Quadrature* PC, and it is simply a discretization of the underlying PIC which is parameterized by neural nets.

# %%
from cirkit.backend.torch.parameters.pic import pc2qpc
pc2qpc(circuit, integration_method="trapezoidal", net_dim=256)

# %% [markdown]
# Let's now inspect again the first layer of ```circuit```.

# %%
print(circuit.nodes[0])
print(circuit.nodes[0].probs().shape)

# %% [markdown]
# We see that the first layer is now parameterized by a neural net (instance of the class ```PICInputNet```), and that its evaluation delivers an equally-shaped tensor as before, i.e. (784, 64, 1, 256).
# 
# Let's check now the second layer.

# %%
print(circuit.nodes[1])
print(circuit.nodes[1].weight().shape)
print(torch.all(torch.isclose(circuit.nodes[1].weight().sum(-1), torch.tensor(1.0))))

# %% [markdown]
# Similarly, the second layer is parameterized by a neural net (instance of the class ```PICInnerNet```), and its evaluation delivers an equally-shaped tensor as before, i.e. (1568, 64, 64). Finally, note that output tensors from neural nets associated to inner layers are already normalized, in the sense that they sum up to 1 over the last dimension, therefore leading to a PC with normalization constant equal to 1.

# %% [markdown]
# That is it, we are done! 🎉 We can now even forget about PICs, and just train as in the [learning a circuit](../learning-a-circuit) notebook as we do next. However, if you want to learn more about PICs (and understand the input arguments of ```pc2qpc```) please check [the original publication](https://arxiv.org/abs/2406.06494).

# %% [markdown]
# ## Learning a Probabilistic (Integral) Circuit

# %%
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Load the MNIST data set and data loaders
transform = transforms.Compose([
    transforms.ToTensor(),
    # Flatten the images and set pixel values in the [0-255] range
    transforms.Lambda(lambda x: (255 * x.view(-1)).long())
])
data_train = datasets.MNIST('datasets', train=True, download=True, transform=transform)
data_test = datasets.MNIST('datasets', train=False, download=True, transform=transform)

# Instantiate the training and testing data loaders
train_dataloader = DataLoader(data_train, shuffle=True, batch_size=256)
test_dataloader = DataLoader(data_test, shuffle=False, batch_size=256)

# Initialize a torch optimizer of your choice,
#  e.g., Adam, by passing the parameters of the circuit
optimizer = optim.Adam(circuit.parameters(), lr=0.005)

# %% [markdown]
# We evaluate our probabilistic circuit on the test data by computing the average log-likelihood and bits per dimension.

print(device)
# %%
with torch.no_grad():
    test_lls = 0.0

    for batch, _ in test_dataloader:
        # The circuit expects an input of shape (batch_dim, num_channels, num_variables),
        # so we unsqueeze a dimension for the channel.
        batch = batch.to(device)

        # Compute the log-likelihoods of the batch
        log_likelihoods = circuit(batch)

        # Accumulate the log-likelihoods
        test_lls += log_likelihoods.sum().item()

    # Compute average test log-likelihood and bits per dimension
    average_ll = test_lls / len(data_test)
    bpd = -average_ll / (28 * 28 * np.log(2.0))
    print(f"Average test LL: {average_ll:.3f}")
    print(f"Bits per dimension: {bpd:.3f}")


