ml PyTorch
ml torchvision
ml OpenCV
ml OpenPose

source ../../wandb/bin/activate

python ../src/conv2d_rnn.py --save_dir='../runs/conv2d/subset100/freeze_vgg-batch200-lr1e-4' --freeze_vgg --lr=1e-4 --use_lr_scheduler --batch_size=200 --gru_hidden_size=64 --subset=100

