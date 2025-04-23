# inbuilt
import os

# 3rd parties
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

# local import
from model.triplet_loss_model import TripletLossModel
from fine_tuning.dataset.triplet_dataset import TripletDataset

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATASET_NAME = 'gen_z_slang'
DATASET_PATH = os.path.join("data", "fine_tuning", f"{DATASET_NAME}_training.jsonl")
MODEL_OUTPUT_PATH = os.path.join("data", "model", f"{MODEL_NAME}-ft-{DATASET_NAME}")


# Load the base model
base_model = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Apply LoRA
lora_config = LoraConfig(
    r=8,                     # Low-rank matrices will have rank 4    
    # lora_alpha=16,    # (default) Low-rank updates will have a strong impact, scaled by 16
    lora_alpha=32,
    lora_dropout=0.1,        # 10% dropout rate for regularization
    # Apply LoRA only to query and value layers in attention
    target_modules=["query", "value"],
    bias="none",             # Do not adapt bias terms
    task_type=TaskType.FEATURE_EXTRACTION  # Use LoRA for feature extraction task
)

lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()

train_dataset = TripletDataset(tokenizer, DATASET_PATH)

model_to_train = TripletLossModel(lora_model)

train_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_PATH,
    per_device_train_batch_size=8,
    learning_rate=2e-4, 
    num_train_epochs=5,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
    fp16=True
)

trainer = Trainer(
    model=model_to_train,
    args=train_args,
    train_dataset=train_dataset,
)
trainer.train()

# trainer.save_model()
model_to_train.base_model.save_pretrained(MODEL_OUTPUT_PATH)


# r=4, lora alpha 32, around 20mins
# {'loss': 0.8388, 'grad_norm': 2.2421302795410156, 'learning_rate': 5.28e-06, 'epoch': 4.48}                                           
# {'loss': 0.8245, 'grad_norm': 2.259315252304077, 'learning_rate': 4.48e-06, 'epoch': 4.56}                                            
# {'loss': 0.8554, 'grad_norm': 1.884563684463501, 'learning_rate': 3.68e-06, 'epoch': 4.64}                                            
# {'loss': 0.7841, 'grad_norm': 2.623325824737549, 'learning_rate': 2.88e-06, 'epoch': 4.72}                                            
# {'loss': 0.7897, 'grad_norm': 2.9021711349487305, 'learning_rate': 2.08e-06, 'epoch': 4.8}                                            
# {'loss': 0.8372, 'grad_norm': 2.0416646003723145, 'learning_rate': 1.28e-06, 'epoch': 4.88}                                           
# {'loss': 0.787, 'grad_norm': 2.171703338623047, 'learning_rate': 4.8e-07, 'epoch': 4.96}                                              
# {'train_runtime': 960.7083, 'train_samples_per_second': 5.199, 'train_steps_per_second': 0.651, 'train_loss': 0.8830189826965332, 'epoch': 5.0}

# r=8, lora alpha 32, learning_rate=2e-4, around 15mins
# {'loss': 0.4902, 'grad_norm': 4.250887393951416, 'learning_rate': 1.472e-05, 'epoch': 4.64}                                           
# {'loss': 0.4312, 'grad_norm': 0.9294217228889465, 'learning_rate': 1.152e-05, 'epoch': 4.72}                                          
# {'loss': 0.3487, 'grad_norm': 0.796898365020752, 'learning_rate': 8.32e-06, 'epoch': 4.8}                                             
# {'loss': 0.3393, 'grad_norm': 2.011441230773926, 'learning_rate': 5.12e-06, 'epoch': 4.88}                                            
# {'loss': 0.3577, 'grad_norm': 4.382317543029785, 'learning_rate': 1.92e-06, 'epoch': 4.96}                                            
# {'train_runtime': 954.9735, 'train_samples_per_second': 5.231, 'train_steps_per_second': 0.654, 'train_loss': 0.5624850849151611, 'epoch': 5.0}