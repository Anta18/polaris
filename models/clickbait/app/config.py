BASE_MODEL_PATH = "models/base_model"
LORA_ADAPTER_PATH = "models/lora_adapter"
MAX_LEN = 256
# Optional small explainer LLM. If you place a small causal LM in this path
# (and optionally its tokenizer), the explainer will use it to produce
# explanations. If the path doesn't exist or loading fails, the code falls
# back to asking the main model (or the heuristic).
EXPLAINER_MODEL_PATH = "models/explainer_model"
