import matplotlib.pyplot as plt
import torchvision

train = torchvision.datasets.MNIST('./data', train=True, download=True)

fig, ax = plt.subplots(6, 6, sharex=True, sharey=True)
for i in range(6):
    for j in range(6):
        ax[i][j].imshow(train.data[6*i+j], cmap="gray")
plt.show()