ml PyTorch
ml torchvision
ml OpenCV
ml OpenPose

source ../../wandb/bin/activate

python ../src/conv2d_rnn.py --save_dir='../runs/conv2d/debug' --use_lr_scheduler --lr=1e-4 --batch_size=100 --gru_hidden_size=64 --subset=100

