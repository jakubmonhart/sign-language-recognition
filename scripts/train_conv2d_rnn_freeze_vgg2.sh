ml PyTorch
ml torchvision
ml OpenCV
ml OpenPose

source ../../wandb/bin/activate

python ../src/conv2d_rnn.py --save_dir='../runs/conv2d_rnn_subset100/freeze-vgg_batch20-lr_schedule' --freeze_vgg=True --lr=1e-2 --batch_size=20 --gru_hidden_size=64 
