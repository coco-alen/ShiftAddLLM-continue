CUDA_VISIBLE_DEVICES=0 python model/llama.py \
    meta-llama/Llama-2-7b-hf \
    --gptq \
    --wbits 3 \
    --groupsize -1 \