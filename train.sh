CUDA_VISIBLE_DEVICES="6" python -m torch.distributed.launch \
--master_port 19996 --nproc_per_node 1 \
train.py