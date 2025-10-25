python -m torch.distributed.launch --nproc_per_node=4 s_00_01_distrib_train_example.py --local_rank <GPU_ID> # GPU_ID: 0, 1, 2, 3

# --nproc_per_node: Specifies the number of GPUs to use per node.
