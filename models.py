import torch
from torchvision import models
from tensorboardX import SummaryWriter

model = models.resnet18(False)

writer = SummaryWriter('./models')
writer.add_graph(model, torch.randn([1,3,64,64]))
writer.close()

