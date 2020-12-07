ml Python
ml binutils/2.32-GCCcore-8.3.0
ml PyTorch
ml torchvision
ml OpenCV
ml OpenPose

python ../src/conv2d_rnn.py --save_dir='../runs/conv2d-rnn-batch100/default' --lr=1e-4 --batch_size=100 --gru_hidden_size=64
