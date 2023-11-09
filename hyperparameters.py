dataset_path = "./CTP_Wires_Chargers_etc"
sample_trial = 50  # keep low to avoid dataloading bottleneck
sample_threshold = 0.1  # sample patches to have more than this standard deviation
crop_size = 86
bit_depth = 8
batch_size = 64
num_workers = 16  # set according to process on node

max_epochs = 100
ckpt_per = 50
base_learning_rate = 1e-10
max_iter = 20

lam = 20
max_loss = 1.8
min_loss = 1.2

debug = True
