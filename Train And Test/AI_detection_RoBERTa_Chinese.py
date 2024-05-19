import os
import re
import json
import torch
import logging
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random

torch.cuda.set_device(2)

# 设置随机种子以确保结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

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
    def __init__(self, texts, labels, tokenizer, max_len=192):
        self.texts = [preprocess_text(text) for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
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
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file :
        for line in file:
            data = json.loads(line)
            if 'human_answers' in data:
                for answer in data['human_answers']:
                    texts.append(preprocess_text(answer))
                    labels.append(0)
            if 'chatgpt_answers' in data:
                for answer in data['chatgpt_answers']:
                    texts.append(preprocess_text(answer))
                    labels.append(1)
    return texts, labels

class EarlyStopping:
    def __init__(self, patience=2, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_wts = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_model_wts)
        else:
            self.best_score = score
            self.best_model_wts = model.state_dict()
            self.counter = 0

def train_model(model, train_loader, val_loader, optimizer, scheduler, scaler, num_epochs=10, patience=2, delta=0.01):
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for batch in train_loader:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            optimizer.zero_grad()
            with autocast():
                outputs = model(**batch)
                loss = outputs[0]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss, val_metrics = evaluate_model(model, val_loader, scaler)
        val_accuracy, val_precision, val_recall, val_f1, val_roc_auc = val_metrics

        logger.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, Val Precision: {val_precision:.4f}, "
              f"Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping")
            break

        scheduler.step(val_loss)

    return model

def evaluate_model(model, data_loader, scaler):
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    data_loader = tqdm(data_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to('cuda') for k, v in batch.items()}
            with autocast():
                outputs = model(**batch)
                loss = outputs[0]
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs[1], dim=-1).cpu().numpy())
                val_labels.extend(batch['labels'].cpu().numpy())
    val_loss /= len(data_loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='binary', zero_division=1)
    val_roc_auc = roc_auc_score(val_labels, val_preds)
    return val_loss, (val_accuracy, val_precision, val_recall, val_f1, val_roc_auc)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, num_labels=2):
    model = RobertaForSequenceClassification.from_pretrained('/home/group2024-detect4/.vscode-server/data/User/RoBERTa_Chinese', num_labels=num_labels)
    model.load_state_dict(torch.load(path))
    model.to('cuda')
    return model

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

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

# 使用提供的最佳超参数
best_params = {'batch_size': 8, 'lr': 1e-05, 'max_len': 128}

# 加载数据
file_path = '/home/group2024-detect4/.vscode-server/data/User/HC3_CN.jsonl'
texts, labels = load_data(file_path)

# 分割数据集
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

# 初始化分词器
tokenizer = AutoTokenizer.from_pretrained('/home/group2024-detect4/.vscode-server/data/User/RoBERTa_Chinese')

# 创建数据集和数据加载器
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=best_params['max_len'])
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=best_params['max_len'])
test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len=best_params['max_len'])

train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

# 初始化模型
roberta_model = RobertaModel.from_pretrained('/home/group2024-detect4/.vscode-server/data/User/RoBERTa_Chinese')
model = CustomRoBERTaModel(roberta_model, num_labels=2, dropout_rate=0.3)
model.classifier.apply(initialize_weights)
model = model.to('cuda')

# 使用正则化的优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-2)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
scaler = GradScaler()

# 训练模型
model = train_model(model, train_loader, val_loader, optimizer, scheduler, scaler)

# 保存模型
save_model(model, '/home/group2024-detect4/.vscode-server/data/User/Detection_Chinese_RoBERTa.pth')

# 评估模型
val_loss, test_metrics = evaluate_model(model, test_loader, scaler)
test_accuracy, test_precision, test_recall, test_f1, test_roc_auc = test_metrics

logger.info(f"Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, "
      f"Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test ROC-AUC: {test_roc_auc:.4f}")
