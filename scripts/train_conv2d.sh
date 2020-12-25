ml PyTorch
ml torchvision
ml OpenCV
ml OpenPose

source ../../wandb/bin/activate

python ../src/conv2d_rnn.py --save_dir='../runs/conv2d/subset2000/batch200-lr1e-4' --lr=1e-4 --use_lr_scheduler --batch_size=200 --gru_hidden_size=256 --subset=2000

