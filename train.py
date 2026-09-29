from transformers import AutoModelForCausalLM
from quantizer import TernaryQuantizer

teacher = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
student = AutoModelForCausalLM.from_pretrained("distilgpt2")

print("Teacher and student loaded")
print("Next step: add tokenizer, dataloader, KD training loop and QAT")

for name, param in student.named_parameters():
    if "weight" in name:
        param.data = TernaryQuantizer.quantize(param.data)

print("Applied starter ternary quantization")
