import torch
from torch.utils.data import DataLoader
from model import get_model
from dataset import PCBDefectDataset

model = get_model().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

train_loader = DataLoader(PCBDefectDataset(), batch_size=16)

for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
