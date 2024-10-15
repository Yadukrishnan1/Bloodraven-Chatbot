import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_NAME = 'HuggingFaceH4/zephyr-7b-beta'

cached_model = None
cached_tokenizer = None

def load_model():
    global cached_model, cached_tokenizer
    if cached_model is None or cached_tokenizer is None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        cached_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config) #
        cached_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return cached_model, cached_tokenizer
