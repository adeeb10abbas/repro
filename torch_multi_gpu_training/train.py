import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # A simple linear layer

    def forward(self, x):
        return self.fc(x)

def generate_synthetic_data(size=1000):
    # Generates synthetic data for demonstration: (input features, targets)
    inputs = torch.randn(size, 10)
    targets = torch.randn(size, 1)
    return TensorDataset(inputs, targets)

def main():
    dist.init_process_group(backend='nccl')

    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Model setup
    model = SimpleModel().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Data setup
    dataset = generate_synthetic_data()
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    model.train()
    for epoch in range(5):  # 5 epochs for simplicity
        sampler.set_epoch(epoch)  # Ensure shuffling different for each epoch
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Rank {local_rank}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()

