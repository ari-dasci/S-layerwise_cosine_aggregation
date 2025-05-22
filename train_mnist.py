import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import MNIST

from models import CNNModel, mnist_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel().to(device)
mnist_train = MNIST(root=".", train=True, download=True, transform=mnist_transforms)
mnist_test = MNIST(root=".", train=False, download=True, transform=mnist_transforms)

writer = SummaryWriter("runs/mnist")

EPOCHS = 10
BATCH_SIZE = 32
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True)
for epoch in range(EPOCHS):
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item()}")


# save model
torch.save(model.state_dict(), "mnist_model.pth")
print("DONE!")

# Write model to tensorboard
x, _ = next(iter(train_loader))
writer.add_graph(model, x.to(device))
writer.close()
