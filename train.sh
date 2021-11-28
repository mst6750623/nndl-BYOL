CUDA_VISIBLE_DEVICES="4,5,7" python -m torch.distributed.launch \
--master_port 19996 --nproc_per_node 3 \
train.py