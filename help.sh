#train
#CUDA_VISIBLE_DEVICES=1 python3 ./train.py --lr=1e-6 --resume='./weights/tb_ICDAR_40000.pth'

#eval
python3 ./eval.py --trained_model='./weights/tb_ICDAR_115000.pth'
