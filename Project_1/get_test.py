
from load_data import FashionNet_Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torchvision.models import vgg16
import torch
from evaluation import compute_avg_class_acc
import torch.nn.functional as F
from utils import converter, plot_loss_acc

test_data = FashionNet_Dataset("FashionDataset/FashionDataset","FashionDataset/FashionDataset/split/test.txt")
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)


Vgg16 = vgg16(pretrained="imagenet")


    # Freeze layers
for param in Vgg16.parameters():
    param.requires_grad = False

    # Modify the model architecture
model = nn.Sequential(
    *list(Vgg16.features.children()),  # Use VGG16 convolutional layers
    nn.Flatten(),  # Flatten the output
    nn.Linear(25088, 4096),  # Add new fully connected layer
    nn.ReLU(inplace=True),  # Add ReLU activation
    nn.Dropout(0.5),  # Add dropout layer
    nn.Linear(4096, 26),  # Add output layer
    nn.Sigmoid() 
    )

model.load_state_dict(torch.load('model_checkpoint.pth'))
prediction_file = "prediction.txt"
with open(prediction_file, "w") as f:
    
    for test_imgs, _ in test_loader:
        test_logits = model.forward(test_imgs.cuda())
        argmax_test_logits = converter(test_logits)
        for row in argmax_test_logits:
                row_str = ' '.join(map(str, row.tolist()))
                f.write(row_str + '\n')
