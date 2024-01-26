# 1. Installed Hugging Face's transformer
# 2. importing libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
# 3. Load pre-trained gpt2 model and tokenizer
model_name="gpt2"
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
model=GPT2LMHeadModel.from_pretrained(model_name)
# 4. Tokenize and Generate Text
def generate_response(prompt, max_length=100, temperature=0.7):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate response
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

user_input=input("My name is Prince and I am your WiseWebWalker. What's Up!")
# user_input="Hey there! what's up?"
response=generate_response(user_input)
print(response)
