# %% [markdown]
# # Learning and Evaluating a Probabilistic Circuit

# %% [markdown]
# In this notebook, we instantiate, learn, and evaluate a probabilistic circuit using ```cirkit```. The probabilistic circuit we build estimates the distribution of MNIST images, which is then evaluated on unseen images, compute marginal probabilities, and sample new images. Here, we focus on the simplest experimental setting, where we want to instantiate a probabilistic circuit for MNIST images using some hyperparameters of our own choice, such as the type of the layers, their size and how to parameterize them. Then, we learn the parameters of the circuit and perform inference using PyTorch.
# 
# A key feature of ```cirkit``` is the _symbolic circuit representation_, which allows us to abstract away from the underlying implementation choices. In the next section, we introduce this symbolic representation and show how to construct a symbolic circuit whose structure and parameterization is tailored for image data sets.

# %% [markdown]
# ## Constructing the Symbolic Circuit

# %% [markdown]
# The **symbolic circuit** is a symbolic abstraction of a tensorized circuit, i.e., a circuit consisting of sum/product/input layers, each grouping several sum/product/input units, respectively. This symbolic representation stores the connections between the layers, the number of units in each layer, and useful metadata about the parameters, such as their shape and parameterization choices. Note that a symbolic circuit does not allocate parameters and cannot be used for learning or inference. By _compiling a symbolic circuit_ using PyTorch, we will later recover a probabilistic circuit that can be learned or be used for inference purposes.
# 
# In ```cirkit.templates```, we provide several templates that can be used to construct symbolic circuits of different structures. In this notebook, we use a high-level template to build a symbolic circuit specifically for image data. To do so, we need to specify some arguments that will possibly yield different architectures and parameterizations. That is, we specify the shape of the images, and select one of the region graphs that exploits the closeness of patches of pixels, such as the _QuadGraph_ region graph.
# See the [notebook on region graphs and sum product layers](../region-graphs-and-parametrisation) for more details about region graphs. Moreover, we select the type of input and inner layers, the number of units within them, and how to parameterize the sum layers. See comments in the code below for more details about each argument.

# %%
from cirkit.templates import data_modalities, utils

symbolic_circuit = data_modalities.image_data(
    (1, 28, 28),                # The shape of MNIST image, i.e., (num_channels, image_height, image_width)
    region_graph='quad-graph',  # Select the structure of the circuit to follow the QuadGraph region graph
    input_layer='categorical',  # Use Categorical distributions for the pixel values (0-255) as input layers
    num_input_units=64,         # Each input layer consists of 64 Categorical input units
    sum_product_layer='cp',     # Use CP sum-product layers, i.e., alternate dense layers with Hadamard product layers
    num_sum_units=64,           # Each dense sum layer consists of 64 sum units
    sum_weight_param=utils.Parameterization(
        activation='softmax',   # Parameterize the sum weights by using a softmax activation
        initialization='normal' # Initialize the sum weights by sampling from a standard normal distribution
    )
)

# %% [markdown]
# We can query some information regarding the symbolic circuit, such as the number of variables it is defined on, and which structural properties it does satisfy.

# %%
# Print some information
print(f'Number of variables: {symbolic_circuit.num_variables}')
print()

# Print which structural properties the circuit satisfies
print(f'Structural properties:')
print(f'  - Smoothness: {symbolic_circuit.is_smooth}')
print(f'  - Decomposability: {symbolic_circuit.is_decomposable}')
print(f'  - Structured-decomposability: {symbolic_circuit.is_structured_decomposable}')

# %% [markdown]
# ## Compiling the Symbolic Circuit with PyTorch

# %% [markdown]
# After we have built our symbolic circuit, it is necessary to **compile** it in order to learn the parameters and perform probabilistic inference. By default, cirkit compiles symbolic circuits using PyTorch 2+. More precisely, by compiling a symbolic circuit, we retrieve a tensorized circuit that specializes [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), thus being very similar to a neural network written in PyTorch. First, we set some random seeds and set the torch device we will use later.

# %%
import random
import numpy as np
import torch

# Set some seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set the torch device to use
device = torch.device('cuda:0')
print(torch.cuda.get_device_name(device))

# %% [markdown]
# Next, we import the ```compile``` function from the ```cirkit.pipeline``` module and compile our symbolic circuit.

# %%%%time
from cirkit.pipeline import compile
circuit = compile(symbolic_circuit)

# %% [markdown]
# Note that the compilation procedure took about three seconds for a circuit with >5700 layers and ~25M parameters, as shown below.

# %%
# Print some statistics
num_layers = len(list(symbolic_circuit.layers))
print(f"Number of layers: {num_layers}")
num_parameters = sum(p.numel() for p in circuit.parameters())
print(f"Number of learnable parameters: {num_parameters}")

# %% [markdown]
# ## Learning a Probabilistic Circuit using PyTorch

# %% [markdown]
# Learning the probabilistic circuit we have compiled above can be done in the same way as any other neural network written using PyTorch. In this notebook, we learn the parameters of the probabilistic circuit as to estimate the distribution of MNIST images. Therefore, below we load the MNIST data set using [torchvision](https://pytorch.org/vision/stable/index.html), and we instantiate the training and testing data loaders. In addition, we select one of the many optimizers implemented in PyTorch, such as [Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html).

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
optimizer = optim.Adam(circuit.parameters(), lr=0.01)

# %% [markdown]
# Finally, we write a classical PyTorch training loop that iterates over batches of MNIST images for some epochs, and optimizes the parameters of the circuit by minimizing the negated average log-likelihood.

# %%
num_epochs = 10
step_idx = 0
running_loss = 0.0
running_samples = 0

# Move the circuit to chosen device
circuit = circuit.to(device)

for epoch_idx in range(num_epochs):
    for i, (batch, _) in enumerate(train_dataloader):
        # The circuit expects an input of shape (batch_dim, num_variables)
        batch = batch.to(device)

        # Compute the log-likelihoods of the batch, by evaluating the circuit
        log_likelihoods = circuit(batch)

        # We take the negated average log-likelihood as loss
        loss = -torch.mean(log_likelihoods)
        loss.backward()
        # Update the parameters of the circuits, as any other model in PyTorch
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.detach() * len(batch)
        running_samples += len(batch)
        step_idx += 1
        if step_idx % 200 == 0:
            average_nll = running_loss / running_samples
            print(f"Step {step_idx}: Average NLL: {average_nll:.3f}")
            running_loss = 0.0
            running_samples = 0

# %% [markdown]
# Similarly, we evaluate our probabilistic circuit on the test data by computing the average log-likelihood and bits per dimension.

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


