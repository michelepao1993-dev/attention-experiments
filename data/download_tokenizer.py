from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
tokenizer.save_pretrained("gemma_tokenizer")