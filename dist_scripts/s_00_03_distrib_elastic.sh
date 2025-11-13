export MASTER_ADDR=localhost
# torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 s_00_03_distrib_elastic.py
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 s_00_03_distrib_elastic.py

