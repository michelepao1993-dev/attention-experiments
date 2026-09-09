from datasets import load_dataset

wikipedia_en = load_dataset(
    "graelo/wikipedia",
    "20230601.en",
    cache_dir="./cache",
    trust_remote_code=True,
)

print(wikipedia_en)

wikipedia_en["train"].to_json(
    "wikipedia_20230601en.json",
    orient="records",
    lines=True,
    num_proc=32,
)