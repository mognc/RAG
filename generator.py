import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("mognc/t5_7_epoch")
model = AutoModelForSeq2SeqLM.from_pretrained("mognc/t5_7_epoch").to(device)

def generate_answer(query, top_passages):
    context = " ".join(top_passages)
    input_text = f"question: {query} context: {context}"
    inputs = tokenizer.encode_plus(
        input_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
