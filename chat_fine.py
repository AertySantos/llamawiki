import sys
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Carregamento do modelo base
base_model_id = "NousResearch/Llama-2-13b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Carregamento do modelo com configuração de quantização
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, quantization_config=bnb_config)

# Carregamento do modelo base sem alterações na configuração de quantização
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Carregamento do tokenizador
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, add_bos_token=True, trust_remote_code=True)

# Carregamento do modelo PeftModel
ft_model = PeftModel.from_pretrained(
    base_model, "llama2-13b-teste-finetune/checkpoint-1000")

while True:
    eval_prompt = input(f"### Answer: ")

    if eval_prompt == 'exit':
        print('Exiting')
        sys.exit()

    # Tokenização da entrada do usuário
    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    # Modo de avaliação do modelo
    ft_model.eval()
    with torch.no_grad():
        # Geração do texto com o modelo treinado
        generated_text = ft_model.generate(
            **model_input, max_new_tokens=400,
            do_sample=True,
            top_p=1,
            temperature=0.01,
            top_k=50,
            output_scores=True,
        )[0]
        decoded_text = tokenizer.decode(
            generated_text, skip_special_tokens=True)
        print(decoded_text)
