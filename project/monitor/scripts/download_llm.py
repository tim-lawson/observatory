from transformers import AutoModelForCausalLM, AutoTokenizer
from util.env import ENV

HF_TOKEN = ENV.HF_TOKEN
MODEL_REPO = "meta-llama/llama-3.1-8b-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForCausalLM.from_pretrained(MODEL_REPO)
