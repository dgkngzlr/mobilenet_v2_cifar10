import torchvision.transforms as T
import torchvision
from torchsummary import summary
from torch.utils.data import DataLoader

from model import *
from train import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = T.Compose([T.ToTensor(),
                       T.Resize(256),
                       T.CenterCrop(224),
                       T.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], std=[0.24703233, 0.24348505, 0.26158768])
                       ])

# All datasets are uniform
train_dataset = torchvision.datasets.CIFAR10("./CIFAR10/train", train=True, \
                                             transform=transform, download=True)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [45000, 5000])

test_dataset = torchvision.datasets.CIFAR10("./CIFAR10/test", train=False, \
                                            transform=transform, download=True)

train_batch_size = 32
val_batch_size = 100
test_batch_size = 100

train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

print("Selected Device :", device)
model = MobileNetV2(n_class=10)
model.to(device)
model.train()

if device == "cpu":
    summary(model, (3, 224, 224), device=device)

else:
    summary(model, (3, 224, 224))

# Shows architecture
# exit(-1)

train_model(device, model, lr=0.05, epochs=150, train_loader=train_loader, val_loader=val_loader, en_schedular=True)
torch.save(model.state_dict(), "MobileNet_v2_cifar10_frozen.pth")

# Test
prec, recall, acc = get_results(device, model, test_loader)
Logger.save_test_info(prec, recall, acc)
