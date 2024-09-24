import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("mognc/t5_7_epoch", use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained("mognc/t5_7_epoch").to(device)

def generate_answer(query, top_passages):
    context = " ".join(top_passages)
    input_text = f"question: {query} context: {context}"
    inputs = tokenizer.encode_plus(
        input_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=2600,
        add_special_tokens=True
    ).to(device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=250,
            num_beams=3,
            length_penalty=2,
            temperature=0.7,
            top_k=50,
            do_sample=True
            )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
