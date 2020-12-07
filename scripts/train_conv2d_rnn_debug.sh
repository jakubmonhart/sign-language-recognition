ml PyTorch
ml torchvision
ml OpenCV
ml OpenPose

source ../../wandb/bin/activate
python ../src/conv2d_rnn.py --save_dir='../runs/conv2d-rnn-debug' --lr=1e-1 --debug_dataset=10 --batch_size=100 --freeze_vgg=True
