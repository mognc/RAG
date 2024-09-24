import torch
from transformers import DPRQuestionEncoderTokenizerFast
from transformers import DPRQuestionEncoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your model and tokenizer
tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-multiset-base")
model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base").to(device)

def encode_query(query):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=128).to(device)
    with torch.no_grad():
        query_embedding = model(**inputs).pooler_output
    return query_embedding.detach().cpu().numpy()
