python commonsense_evaluate.py \
  --model LLaMA-7B \
  --adapter LoRA \
  --dataset piqa \
  --base_model yahma/llama-7b-hf \
  --batch_size 1 \
  --lora_weights ./trained_models/llama-lora-