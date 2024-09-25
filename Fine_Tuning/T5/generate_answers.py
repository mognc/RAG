from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("mognc/t5_7_epoch")
model = AutoModelForSeq2SeqLM.from_pretrained("mognc/t5_7_epoch")


def generate_answer(query):
    input_text = f"Question: {query}"
    input = tokenizer.encode_plus(
        input_text,
        max_length=300,
        truncation=True,
        padding=True,
        add_special_tokens=True,
        return_tensors='pt'
    )
    input_ids = input['input_ids']
    attention_mask = input['attention_mask']

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=512,
        num_beams=9,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def generate_answer_context(query, context):
    input_text = f"Question: {query} Context: {context}"
    input = tokenizer.encode_plus(
        input_text,
        max_length=3000,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
        return_tensors='pt'
    )

    input_ids = input['input_ids']
    attention_mask = input['attention_mask']

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=512,
        num_beams=9,
        early_stopping=True,
        temperature=1.0,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
