from transformers import TextStreamer
from model import init_model



model, tokenizer = init_model()
# Helper function for inference
def do_gemma_3n_inference(messages, max_new_tokens = 2000):
    print("=" * 60)
    print("GEMMA 3N MODEL INFERENCE")
    print("=" * 60)
    _ = model.generate(
        **tokenizer.apply_chat_template(
            messages,
            add_generation_prompt = True, # Must add for generation
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
        ).to("cuda"),
        max_new_tokens = max_new_tokens,
        temperature = 1.0, top_p = 0.95, top_k = 1000,
        streamer = TextStreamer(tokenizer, skip_prompt = True),
    )


