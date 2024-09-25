import pandas as pd
import json
import torch
from datasets import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

with open('RAG\datasets\Game_of_Thrones_T5_FineTuning.txt', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['input_text'] = df.apply(lambda x: f"Question: {x['question']} Context: {x['context']}", axis=1)
df['target_text'] = df['answer']
dataset = Dataset.from_pandas(df[['input_text', 'target_text']])


def preprocess_function(examples):
    inputs = examples['input_text']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    labels = tokenizer(examples["target_text"], max_length=512, truncation=True, padding="max_length", return_tensors="pt")
    
    return {
        "input_ids": model_inputs["input_ids"].squeeze(),
        "attention_mask": model_inputs["attention_mask"].squeeze(),
        "labels": labels["input_ids"].squeeze(),
    }

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=7,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
trainer.save_model('./flan_t5_finetuned')
tokenizer.save_pretrained('./flan_t5_finetuned')