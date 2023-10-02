import os
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm, trange
from utils import *
import networks as models
import networks_new as models_new
import wandb
import argparse
import torch.nn.functional as F
from einops import rearrange
import timeit

parser = argparse.ArgumentParser(description= "Image classification usning Complex valued network ")
parser.add_argument('--batchsize', type= int, default= 16)
parser.add_argument('--learning_rate', '-lr', type= float, default= 1e-4, help='Learning rate')
parser.add_argument('--max_epochs', type= int, default= 50, help= 'Maximum number of epochs')
parser.add_argument('--model', type= str, default='Alexnet')
parser.add_argument('--classes', type= int, default= 10, help= 'Number of classes in the dataset')
parser.add_argument('--project_name', type= str, default= ' new Loss CIFAR10', help= 'Name of the wandb project')


args = parser.parse_args()

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# login with wandb
wandb.login()

def criterion2(y, theta):
    y = F.one_hot(y, num_classes= 10)
    loss = y * theta
    return loss.mean()

# define model pipeline
def model_pipeline(hyperparameters):

    # tell wandb to get started
    run = wandb.init(project= args.project_name, config= hyperparameters, name= args.model, save_code= True)

    config = run.config


    model, train_loader, test_loader, criterion, optimizer = make()

    run.watch(model)

    # variables for storing losses during one epoch
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    best_test_acc = 0.0

    # define schedulers here
    ######################################
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= 200)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor= 0.1, patience= 5, verbose= True)
    


    for epoch in (t := trange(config.max_epochs)):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, config)

        test_loss, test_acc = test(model, test_loader, criterion, config)

        # scheduler step
        ###########################
        scheduler1.step()
        scheduler2.step(test_loss)

        if test_acc > best_test_acc:
            print(f"Saving ... ")
            state = {
                    'model': model.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                    }

            path = os.path.join('saved_models/cifar10', config.model)

            # if not os.path.isdir(path):
                # os.mkdir(path)

            # torch.save(state, os.path.join(path, "saved_model.pth"))
            best_test_acc = test_acc

            ######################
            # use config.model to make a path and store the model there
            ######################

        # Now, print the information per epoch
        # Use t.set_description() for updating the values
        t.set_description(f"Epoch: {epoch + 1} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Valid Loss: {test_loss:.4f} Valid Acc: {test_acc:.4f} Best Acc: {best_test_acc:.4f}")

        wandb.log({
            "Epoch": epoch + 1,
            "Train Loss": train_loss,
            "Train Accuracy": train_acc,
            "Test Loss": test_loss,
            "Test Accuracy": test_acc,
            "Best Accuracy": best_test_acc,
            })

    # save the model in exchangable ONNX format
    # For saving the model in onnx format, we need a dummy input
    # dummy_input = torch.randn(4, 3, 50, 50, device= device).type(torch.complex64)
    # store the model
    # path_onnx = os.path.join(path, "model.onnx")
    # torch.onnx.export(model, dummy_input, path_onnx)
    # wandb.save(path_onnx)
    # wandb.save(path_onnx)
    wandb.save(os.path.join(path, config.model, "saved_model.pth"))

    run.finish()

    return model


def make():
    # get data set
    train, test = get_train_data(), get_test_data()
    train_loader = make_loader(train, batch_size= args.batchsize)
    test_loader = make_loader(test, batch_size= args.batchsize)

    # assign model here.
    # model = model.args.model(num_classes= 10)
    model = models.CDS_E(num_classes= 10).to(device)
    # model = models.AlexNet(num_classes= 10).to(device)
    # model = models.VGG('Vgg11', num_classes= 10).to(device)
    # model = models.resnet18(num_classes= 10).to(device)
    # model = models_new.resnet18(num_classes=10).to(device)
    # print(model)

    # model = models.CDS_E(num_classes= 10).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(
            model.parameters(), lr= args.learning_rate, weight_decay= 5e-4)
    # optimizer = torch.optim.sgd(
    #         model.parameters(), lr= args.learning_rate, momentum= 0.8, weight_decay= 5e-4
    #         )
            

    return model, train_loader, test_loader, criterion, optimizer


def get_train_data():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(58),
        transforms.CenterCrop(56),
        transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(10),
        ToHSV(),
        ToComplex(),
        # ToiRGB(),
    ])

    trainset = torchvision.datasets.CIFAR10(
            root= 'data', train= True, download= True, transform= transform_train
            )

    return trainset

def get_test_data():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(56),
        ToHSV(),
        ToComplex(),
        # ToiRGB(),
        ])

    testset = torchvision.datasets.CIFAR10(
            root= 'data', train= False, download= True, transform= transform_test  
            )

    return testset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
            dataset= dataset, batch_size= batch_size, shuffle= True, pin_memory= True, num_workers= 4
            )
    return loader

def train(model, train_loader, criterion1, optimizer, config):
    # put the model in training phase
    model.train()

    run_loss = 0.0
    cnt = 0
    total = 0.0
    correct = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()


        # start = timeit.default_timer()
        outputs = model(images)
        # end = timeit.default_timer()
        # print(f"Time taken to process input: {end-start}")
        outputs = rearrange(outputs, 'b c h w -> b (c h w)')
        assert outputs.dtype == torch.complex64
        outputs_phase = outputs.angle()
        outputs_magnitude = outputs.abs()
        # print(f"Min value of phase: {outputs_phase.min()}")

        # print(f"Shape of outputs_magnitude: {outputs_magnitude.shape}")

        loss1= criterion1(outputs_magnitude, labels)
        loss2 = criterion2(labels, outputs_phase)
        lamb = 1
        loss = (loss1 - lamb*loss2)/ 2

        run_loss += loss.item()
        total += labels.size(0)
        cnt += 1
        _, predicted = torch.max(outputs_magnitude.data, 1)
        correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(train_loader),'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            %(run_loss / (batch_idx + 1), 100.*correct/total, correct, total))

    return run_loss / cnt, correct / total



def test(model, test_loader, criterion1, config):
    # put the model in evaluation mode
    model.eval()

    with torch.no_grad():
        correct, total, cnt = 0, 0, 0

        run_loss = 0.0
        pr_time = 0.0

        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            start = timeit.default_timer()
            outputs = model(images)
            end = timeit.default_timer()
            pr_time += (end-start)
            outputs = rearrange(outputs, 'b c h w -> b (c h w)')
            assert outputs.dtype == torch.complex64
            outputs_phase = outputs.angle()
            outputs_magnitude = outputs.abs()

            # print(f"Shape of outputs_magnitude: {outputs_magnitude.shape}")

            loss1= criterion1(outputs_magnitude, labels)
            loss2 = criterion2(labels, outputs_phase)
            # lamb = math.pi / 360
            # loss = (loss1 + lamb*loss2)/ 2
            loss = (loss1 + loss2)
            run_loss += loss
            _, predicted = torch.max(outputs_magnitude.data, 1)
            total += labels.size(0)
            cnt += 1
            correct += (predicted == labels).sum().item()
            progress_bar(batch_idx, len(test_loader),'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                %(run_loss / (batch_idx + 1), 100.*correct/total, correct, total))
        print(f"Average time for inference: {pr_time / cnt}")


    return run_loss / cnt, correct / total


model = model_pipeline(args)
