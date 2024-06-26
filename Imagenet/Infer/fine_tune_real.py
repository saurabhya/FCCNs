import torch
import networks as models
from tqdm import tqdm, trange
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from utils import *
import argparse
import torch.nn.functional as F
from einops import rearrange
import timeit
import complexnn as comp
from torch.utils.data import random_split, Dataset


class Office31:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=root_dir, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def freeze_layers(model):
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True


def count_param(model):
    return sum(p.numel() for p in model.parameters())


class NewDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx,
                             cls_name in enumerate(self.classes)}
        self.img_paths = self._get_img_paths()

    def _get_img_paths(self):
        img_paths = []
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                img_paths.append((img_path, cls_name))
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, cls_name = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[cls_name]


parser = argparse.ArgumentParser(description="Finetune trained model")
parser.add_argument(
    "-a", "--arch", metavar="ARCH", default="resnet18", help="model architecture"
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
parser.add_argument(
    "--pretrained", default="model_best.pth.tar", help="location of pretrained weight"
)
parser.add_argument(
    "--epochs", default=90, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "-b",
    "--batchsize",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256), this is the total "
    "batch size of all GPUs on the current node when "
    "using Data Parallel or Distributed Data Parallel",
)
parser.add_argument(
    "--lr",
    "--learning_rate",
    default=1e-1,
    type=float,
    metavar="LR",
    help="initial learning rate",
    dest="lr",
)
parser.add_argument("--momentum", default=0.9, type=float,
                    metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight_decay",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
    dest="weight_decay",
)
parser.add_argument(
    "--classes", type=int, default=10, help="Number of classes in the dataset"
)


args = parser.parse_args()


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def criterion2(y, theta):
    y = F.one_hot(y, num_classes=31)
    loss = y * theta
    return loss.mean()


# define model pipeline
def model_pipeline(hyperparameters):
    model, train_loader, test_loader, criterion, optimizer = make(args)

    # variables for storing losses during one epoch
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []
    best_test_acc = 0.0

    # define schedulers here
    ######################################
    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.1, patience=5, verbose=True
    )

    for epoch in (t := trange(args.epochs)):
        train_loss, train_acc = train(
            model, train_loader, criterion, optimizer)

        test_loss, test_acc = test(model, test_loader, criterion)

        # scheduler step
        ###########################
        scheduler1.step()
        scheduler2.step(test_loss)

        if test_acc > best_test_acc:
            print(f"Saving ... ")
            state = {
                "model": model.state_dict(),
                "acc": test_acc,
                "epoch": epoch,
            }

            # torch.save(state, os.path.join(path, "saved_model.pth"))
            best_test_acc = test_acc

            ######################
            # use config.model to make a path and store the model there
            ######################

        # Now, print the information per epoch
        # Use t.set_description() for updating the values
        t.set_description(
            f"Epoch: {epoch + 1} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Valid Loss: {test_loss:.4f} Valid Acc: {test_acc:.4f} Best Acc: {best_test_acc:.4f}"
        )

    return model


def make(args):
    # get data set
    train, test = get_train_data()
    train_loader = make_loader(train, batch_size=args.batchsize)
    test_loader = make_loader(test, batch_size=args.batchsize)

    # assign model here.
    if args.arch == "resnet18":
        model = models.resnet18(num_classes=1000)
    elif args.arch == "resnet50":
        model = models.resnet50(num_classes=1000)
    elif args.arch == "resnet152":
        model = torchvision.models.resnet152(pretrained=True)

    else:
        model = None
        print("Model name in correct")

    # change last layer of the model
    model.fc = nn.Linear(4 * 512, args.classes)

    print(f"Model parameters: {count_param(model)}")

    model = model.to(device)

    freeze_layers(model)

    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    return model, train_loader, test_loader, criterion, optimizer


def get_train_data():
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomRotation(10),
            # ToHSV(),
            # ToComplex(),
            # ToiRGB(),
        ]
    )

    # dataset = Office31(root_dir="Office_31/webcam/images",
    #                    transform=transform_train)
    trainset = NewDataset(
        root_dir="TMD_SIBGRAP-2021/NewR22/train", transform=transform_train
    )
    testset = NewDataset(
        root_dir="TMD_SIBGRAP-2021/NewR22/test", transform=transform_train
    )

    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    #
    # trainset, testset = random_split(dataset, [train_size, test_size])
    print(f"length of trainset: {len(trainset)}")
    print(f"length of testset: {len(testset)}")

    return trainset, testset


# def get_test_data():
#     transform_test = transforms.Compose(
#         [transforms.ToTensor(), transforms.Resize(56), ToHSV(), ToComplex(), ToiRGB()]
#     )
#
#     testset = torchvision.datasets.CIFAR10(
#         root="data", train=False, download=True, transform=transform_test
#     )
#
#     return testset


def make_loader(dataset, batch_size):
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    return loader


def train(model, train_loader, criterion1, optimizer):
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
        # print(f"Min value of phase: {outputs_phase.min()}")

        # print(f"Shape of outputs_magnitude: {outputs_magnitude.shape}")

        loss = criterion1(outputs, labels)

        run_loss += loss.item()
        total += labels.size(0)
        cnt += 1
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

        # progress_bar(
        #     batch_idx,
        #     len(train_loader),
        #     "Loss: %.3f | Acc: %.3f%% (%d/%d)"
        #     % (run_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        # )

    return run_loss / cnt, correct / total


def test(model, test_loader, criterion1):
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
            pr_time += end - start

            loss = criterion1(outputs, labels)
            run_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            cnt += 1
            correct += (predicted == labels).sum().item()
            # progress_bar(
            #     batch_idx,
            #     len(test_loader),
            #     "Loss: %.3f | Acc: %.3f%% (%d/%d)"
            #     % (run_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            # )
        print(f"Average time for inference: {pr_time / cnt}")

    return run_loss / cnt, correct / total


model = model_pipeline(args)
