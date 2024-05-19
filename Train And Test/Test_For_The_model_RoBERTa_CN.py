import torch
import re
import logging
from transformers import AutoTokenizer, RobertaModel
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_text(text):
    if text is None:
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = [preprocess_text(text) for text in texts]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long)
        }

class CustomRoBERTaModel(nn.Module):
    def __init__(self, roberta, num_labels=2, dropout_rate=0.3):
        super(CustomRoBERTaModel, self).__init__()
        self.roberta = roberta
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(roberta.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return (loss, logits)

def load_model(path, num_labels=2):
    roberta_model = RobertaModel.from_pretrained('RoBERTa_Chinese')
    model = CustomRoBERTaModel(roberta_model, num_labels=num_labels, dropout_rate=0.3)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()
    return model

def predict(text, model, tokenizer, max_len=128):
    dataset = TextDataset([text], tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to('cpu') for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs[1]
            probs = torch.softmax(logits, dim=1)
            ai_prob = probs[0][1].item()
            human_prob = probs[0][0].item()
            return ai_prob, human_prob

if __name__ == "__main__":
    model_path = 'Detection_Chinese_RoBERTa.pth'
    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained('RoBERTa_Chinese')

    while True:
        user_input = input("请输入一段文字进行检测 (输入'q'退出)：")
        if user_input.lower() == 'q':
            break

        ai_prob, human_prob = predict(user_input, model, tokenizer)
        print(f"AI生成的概率: {ai_prob:.4f}, 人类生成的概率: {human_prob:.4f}")

        if ai_prob > human_prob:
            print("该文字很可能是由AI生成的。")
        else:
            print("该文字很可能是由人类生成的。")
