# command for running the code
CUDA_VISIBLE_DEVICES=5 python main.py --arch resnet50 -b 128 --lr 0.01 --resume model_best.pth.tar --epochs 120 ../imagenet
