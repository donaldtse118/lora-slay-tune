
# 🧠 lora-slay-tune 🔥  
*Fine-tuning language models to finally "get" Gen Z.*

## 🚀 Overview

**lora-slay-tune** is a LoRA-based fine-tuning project built to make sentence embeddings *slay* with Gen Z slang. This repo uses triplet loss to train an encoder that can better understand the semantic similarity between phrases that would otherwise leave older models (and older humans) confused. Think of it as giving your model some ✨riz✨.

> 🧑‍💻 Inspired by the daily chaos of decoding Gen Z lingo from my son. This repo is a tribute to all the confused dads out there.

## 🧩 Key Features

- 🔧 LoRA applied to `all-MiniLM-L6-v2` encoder
- 💡 Uses **triplet loss** for semantic alignment
- 🗣️ Tailored training dataset with Gen Z slang phrases
- 🔥 Outputs embeddings that "get it" when it comes to slang similarity



## 🧠 Training

We fine-tune using Hugging Face's `Trainer`, with `LoRA` for parameter-efficient tuning and `TripletMarginLoss` to align embeddings of semantically close phrases.

### Sample Config

```python
lora_config = LoraConfig(
    r=8, # I used 8, tried r=4 can't learn Gen Z well, might be it is too difficult
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"],
    task_type=TaskType.FEATURE_EXTRACTION
)

training_args = TrainingArguments(
    learning_rate=2e-4,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    output_dir="output/lora-slay-tune",
    logging_steps=10,
    fp16=True
)
```

## 🧐 Why Triplet Loss?

Triplet loss helps the model learn *relative* similarities between embeddings:  
- `"This movie was lit"` should be closer to `"That film slapped"`  
- and far away from `"It was boring af"`.

## 📊 Example Use Case

> Given a pair of Gen Z phrases, the model will return whether they hit the same vibe or not.

### 1. Fine-tuning the Model
Fine-tune the model with your own Gen Z slang data by adding or updating the `gen_z_slang_training.jsonl` file.

- **Anchor**: Input Gen Z slang (the two phrases you want to compare).
- **Positive pair**: Gen Z phrases that are similar in meaning or tone (e.g., "lit" and "fire").
- **Negative pair**: Gen Z phrases that are not similar (e.g., "sus" and "gravy train").

### 2. Evaluation
Update `gen_z_slang_evaluation.jsonl` in the same format and execute `evaluate.py` to compare the performance of the base model and your fine-tuned model.


## 🧐 Evaluation

To assess the effectiveness of our fine-tuned model, we ran an evaluation comparing the base model and the fine-tuned model on a set of 200 Gen Z slang phrase pairs.

### Evaluation Results:

- **Dataset**: 200 rows of Gen Z slang phrase pairs
- **Correct Samples**:
  - **Before Fine-Tuning**: Avg. distance = **1.4208**
  - **After Fine-Tuning**: Avg. distance = **0.8989**
  - **Distance Change**: **-0.5219** (a **-36.74%** decrease)

- **Wrong Samples**:
  - **Before Fine-Tuning**: Avg. distance = **1.4919**
  - **After Fine-Tuning**: Avg. distance = **1.9650**
  - **Distance Change**: **+0.4731** (a **32.01%** increase)

### What’s the vibe?
- The fine-tuned model did **way better** at pulling closer together similar Gen Z phrases (a **-36.74%** reduction in distance).
- But, for the incorrect pairs, it slipped a bit and pulled them even further apart (**+32.01%** increase in distance).

This shows that while fine-tuning has a clear win for similar pairs, there's still room to dial it in for cases that aren't so similar. Keep tuning, keep slaying!

## 🤝 Contributing

PRs welcome. Especially from any Gen Z-er who wants to explain why "mid" is an insult.

## 📜 License

MIT. Go wild, but don’t ghost us 🫠
