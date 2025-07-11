from unsloth import FastModel
import torch
import yaml

with open('config_unsloth.yaml', 'r') as file:
    config = yaml.safe_load(file)







def init_model():
    model, tokenizer = FastModel.from_pretrained(
        model_name =config["model"]["model_name"],
        dtype = config["model"]["dtype"], # None for auto detection
        max_seq_length = config["model"]["max_seq_length"], # Choose any for long context!
        load_in_4bit = config["model"]["load_in_4bit"],  # 4 bit quantization to reduce memory
        full_finetuning = config["model"]["full_finetuning"], # [NEW!] We have full finetuning now!
        token = config["model"]["token"], # use one if using gated models
    )
    return model, tokenizer



if __name__ == "__main__":
    model, tokenizer = init_model()
    print(model)
    print(tokenizer)



