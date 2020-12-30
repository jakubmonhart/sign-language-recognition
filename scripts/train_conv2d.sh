ml PyTorch
ml torchvision
ml OpenCV
ml tqdm

source ../../wandb/bin/activate

python ../src/conv2d_rnn.py --save_dir='../runs/conv2d/subset100/batch5-lr1e-4' --lr=1e-4 --use_lr_scheduler --batch_size=5 --gru_hidden_size=64 --subset=100

