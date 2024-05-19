import fitz  # PyMuPDF
import re
import torch
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
    def __init__(self, texts, tokenizer, max_len=256):
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

def load_model(path, model_name):
    roberta_model = RobertaModel.from_pretrained(model_name)
    model = CustomRoBERTaModel(roberta_model, num_labels=2, dropout_rate=0.3)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.to('cpu')
    model.eval()
    return model

def predict(text, model, tokenizer, max_len=256):
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

class PDFReader:
    def __init__(self, file_path, language, output_path, model_path, tokenizer_name):
        self.file_path = file_path  # PDF文件路径
        self.language = language  # 语言类型（中文 'zh' 或 英文 'en'）
        self.output_path = output_path  # 保存文件的路径
        self.doc = fitz.open(file_path)  # 打开PDF文件
        self.model_path = model_path
        self.tokenizer_name = tokenizer_name
        self.model = load_model(model_path, tokenizer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def read_pdf(self):
        text = ""
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            text += page.get_text()  # 获取每一页的文本内容
        return text

    def split_text(self, text):
        if self.language == 'zh':
            return self.split_text_chinese(text)  # 中文分段
        elif self.language == 'en':
            return self.split_text_english(text)  # 英文分段

    def split_text_chinese(self, text):
        segments = []
        current_segment = ""
        for char in text:
            current_segment += char
            if len(current_segment) >= 100 and char in "。！？":  # 满足字数且遇到句子分隔符时
                segments.append(current_segment)
                current_segment = ""
        if current_segment:
            segments.append(current_segment)  # 添加最后一段
        return segments

    def split_text_english(self, text):
        segments = []
        current_segment = ""
        words = re.split(r'\s+', text)
        word_count = 0
        for word in words:
            current_segment += word + " "
            word_count += 1
            if word_count >= 100 and re.match(r'[.!?]', word):  # 满足单词数且遇到句子分隔符时
                segments.append(current_segment.strip())
                current_segment = ""
                word_count = 0
        if current_segment:
            segments.append(current_segment.strip())  # 添加最后一段
        return segments

    def filter_text(self, segment):
        return re.sub(r'[\n\r\t]', ' ', segment)  # 去除特殊符号

    def evaluate_text(self, text):
        ai_prob, human_prob = predict(text, self.model, self.tokenizer)
        return ai_prob

    def highlight_pdf(self):
        for page_num in range(len(self.doc)):
            page = self.doc.load_page(page_num)
            text = page.get_text()
            segments = self.split_text(text)
            for segment in segments:
                filtered_segment = self.filter_text(segment)
                probability = self.evaluate_text(filtered_segment)
                print(f"Segment: {filtered_segment[:30]}..., Probability: {probability:.4f}")  # 调试信息，只显示前30个字符
                if 0.2 < probability < 0.6:
                    color = (1, 1, 0)  # 黄色
                elif probability >= 0.6:
                    color = (1, 0, 0)  # 红色
                else:
                    continue
                highlight = page.search_for(filtered_segment)
                if not highlight:
                    print(f"No highlight found for segment: {filtered_segment[:30]}...")  # 调试信息
                    continue
                print(f"Highlight found for segment: {filtered_segment[:30]}...")  # 调试信息
                for inst in highlight:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=color)
                    highlight.update()
        self.doc.save(self.output_path)  # 使用指定的路径保存文件

# 使用示例
file_path = 'C:\\Users\\Chenx\\Desktop\\7.pdf'
output_path = 'C:\\Users\\Chenx\\Desktop\\8.pdf'
language = 'zh'  # 或 'zh'

if language == 'zh':
    model_path = 'Detection_Chinese_RoBERTa.pth'
    tokenizer_name = 'RoBERTa_Chinese'
else:
    model_path = 'Detection_English_RoBERTa_try_1.pth'
    tokenizer_name = 'RoBERTa_English'

reader = PDFReader(file_path, language, output_path, model_path, tokenizer_name)
reader.highlight_pdf()
