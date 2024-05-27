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




def main(args):

    test_data = FashionNet_Dataset("FashionDataset/FashionDataset","FashionDataset/FashionDataset/split/test.txt")
    val_data = FashionNet_Dataset("FashionDataset/FashionDataset","FashionDataset/FashionDataset/split/val.txt")
    # train_data = FashionNet_Dataset("FashionDataset/FashionDataset","FashionDataset/FashionDataset/split/train.txt")
    train_data = FashionNet_Dataset("FashionDataset/FashionDataset","FashionDataset/FashionDataset/split/train_sorted2.txt")
    print(train_data.__len__())
    print(train_data.count_classes())
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

    
    stats = ((0.4054, 0.3889, 0.3523), (0.1606, 0.1585, 0.1616))
    
    aug_tfms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.RandomRotation(30),
                            transforms.RandomHorizontalFlip(0.8),  
                            transforms.RandomCrop(224, padding=4),  
                            transforms.Normalize(*stats, inplace=True),
                            ])
    # aug_tfms_2 = transforms.Compose([
    #                         transforms.ColorJitter(contrast=0.5),
    #                         transforms.RandomRotation(30),
    #                         transforms.CenterCrop(200),
    #                         ])
    
    augmented_dataset = FashionNet_Dataset("FashionDataset/FashionDataset","FashionDataset/FashionDataset/split/train_aug.txt")
    # augmented_dataset_2 = FashionNet_Dataset("FashionDataset/FashionDataset","FashionDataset/FashionDataset/split/train.txt")
    augmented_dataset.transform = aug_tfms
    # augmented_dataset_2.transform = aug_tfms_2

    combined_dataset = torch.utils.data.ConcatDataset([train_data,augmented_dataset])
    
    
    train_loader = DataLoader(combined_dataset , batch_size=256, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)


    # Print the model summary
    print(model)
    model.cuda()



    weight = torch.tensor([0.0140807651,0.0443676939,0.181323061,0.0560573858,0.519872476 ,0.0144792774,0.569819341
                           ,0.474442083,0.182917109,0.742640808,
                           0.887300744,0.210414453,0.102284803
                          ,0.152364506,0.618304995,0.626806589,0.00252391073
                          ,0.0662858661,0.01434644, 0.00690754516,0.454091392,0.913868225,0.0445005313,
                          0.0589798087,0.347848034,0.893172157
                        ]) # higher weight for positive class (has less numbers)
    criterion = nn.BCEWithLogitsLoss(weight=weight).cuda()


    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    stat_training_loss = []
    stat_val_loss = []
    stat_training_acc = []
    stat_val_acc = []

    
    for epoch in range(args.epochs):
        training_loss = 0
        training_acc = 0
        training_samples = 0
        val_loss = 0
        val_acc = 0
        val_samples = 0
        batch_number_training = 0
        batch_number_val = 0
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.cuda()
            labels = labels.cuda()
            labels = labels.cuda().squeeze(1)
            batch_size = imgs.shape[0]
            optimizer.zero_grad()
            logits = model.forward(imgs)

            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            argmax_labels = converter(labels)
            argmax_logits = converter(logits)
            training_acc += compute_avg_class_acc(argmax_labels.cuda(), argmax_logits)
            training_loss += batch_size * loss.item()
            training_samples += batch_size
            batch_number_training +=1
            # validation

        model.eval()
        

        for val_imgs, val_labels in valid_loader:
            val_labels = val_labels.cuda().squeeze(1)
            batch_size = val_imgs.shape[0]
            val_logits = model.forward(val_imgs.cuda())
            loss = criterion(val_logits, val_labels.float().cuda())

            argmax_val_labels = converter(val_labels)
            argmax_val_logits = converter(val_logits)

            val_acc += compute_avg_class_acc(argmax_val_labels.cuda(), argmax_val_logits)
            val_loss += batch_size * loss.item()
            val_samples += batch_size
            batch_number_val += 1
            # update stats

            
        stat_training_loss.append(training_loss/training_samples)
        stat_val_loss.append(val_loss/val_samples)
        stat_training_acc.append(training_acc/batch_number_training)
        stat_val_acc.append(val_acc/batch_number_val)
        # print
        print(training_samples)
        print(f"Epoch {(epoch+1):d}/{args.epochs:d}.. Learning rate:depends.. Train loss: {(training_loss/training_samples):.4f}.. Train acc: {(training_acc/batch_number_training):.4f}.. Val loss: {(val_loss/val_samples):.4f}.. Val acc: {(val_acc/batch_number_val):.4f}")

        
    
        prediction_file = "prediction.txt"
        plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, args.fig_name)
        with open(prediction_file, "w") as f:
            if args.test:
                for test_imgs, _ in test_loader:
                    test_logits = model.forward(test_imgs.cuda())
                    argmax_test_logits = converter(test_logits)
                    # Printing the first row of size [1, 6] for each of the 128 rows
                    for row in argmax_test_logits:
                        row_str = ' '.join(map(str, row.tolist()))
                        f.write(row_str + '\n')
                    

        checkpoint_path = 'model_checkpoint.pth'


        checkpoint = {
            'model_state_dict': model.state_dict(),
        }

        # Save the checkpoint
        torch.save(checkpoint, checkpoint_path)

    






if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--fig_name',type=str, help='')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()
    print(args)
    main(args)