ml PyTorch
ml torchvision
ml OpenCV
ml OpenPose

source ../../wandb/bin/activate

python ../src/conv2d_rnn.py --save_dir='../runs/conv2d/subset300/freeze_vgg-batch100-lr1e-4' --freeze_vgg=True --lr=1e-4 --batch_size=100 --gru_hidden_size=64 --subset=300
