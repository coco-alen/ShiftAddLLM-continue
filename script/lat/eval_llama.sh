CUDA_VISIBLE_DEVICES=0 python model/llama.py \
    meta-llama/Llama-2-7b-hf \
    --wbits 3 \
    --groupsize 128 \
    --lat \
    --bcq_round 50 \
    --act_quant_int 4 \
    --act_quant_per_block

    # --temp_storage ./weight \