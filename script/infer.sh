# CUDA_VISIBLE_DEVICES=3 python model/opt.py \
#     facebook/opt-125m \
#     --wbits 3 \
#     --lat \
#     --load_temp_storage ./weight \
#     --infer_kernel \
#     --benchmark 128

CUDA_VISIBLE_DEVICES=0 python model/opt.py \
    facebook/opt-2.7b \
    --act_quant_int 8 \
    --load /home/hanrui/projects/yipin/ShiftAddLLM-continue/opt2.7b-w3-lat.pt