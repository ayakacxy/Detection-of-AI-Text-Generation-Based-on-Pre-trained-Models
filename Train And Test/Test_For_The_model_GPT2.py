import re
import torch
import logging
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn


torch.cuda.set_device(3)
# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class GLTRGPT2Model(nn.Module):
    def __init__(self, gpt2, num_labels=2, dropout_rate=0.3):
        super(GLTRGPT2Model, self).__init__()
        self.gpt2 = gpt2
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(gpt2.config.n_embd, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1][:, -1, :]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return (loss, logits)

def load_model(model_path, tokenizer_path, device='cuda'):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained(tokenizer_path)
    model = GLTRGPT2Model(gpt2_model, num_labels=2, dropout_rate=0.3)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model, tokenizer

def predict(model, tokenizer, text, max_len=256, device='cuda'):
    text = preprocess_text(text)
    inputs = tokenizer(text, max_length=max_len, padding='max_length', truncation=True, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        _, logits = model(**inputs)
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
    return probabilities

def main():
    model_path = '/home/group2024-detect4/.vscode-server/data/User/Detection_English_GPT2.pth'
    tokenizer_path = '/home/group2024-detect4/.vscode-server/data/User/GPT2_English'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, tokenizer = load_model(model_path, tokenizer_path, device)

    while True:
        text = input("请输入一句话（或输入'退出'结束）：")
        if text.lower() == '退出':
            break
        probabilities = predict(model, tokenizer, text, device=device)
        human_prob, ai_prob = probabilities[0]
        print(f"这句话由AI生成的概率：{ai_prob:.4f}")
        print(f"这句话由人类生成的概率：{human_prob:.4f}")

if __name__ == "__main__":
    main()
