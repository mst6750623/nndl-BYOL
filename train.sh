CUDA_VISIBLE_DEVICES="6" python -m torch.distributed.launch \
--master_port 19995 --nproc_per_node 1 \
train.py