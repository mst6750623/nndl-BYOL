CUDA_VISIBLE_DEVICES="0,2,3,5,6" python -m torch.distributed.launch \
--master_port 19996 --nproc_per_node 5 \
train.py