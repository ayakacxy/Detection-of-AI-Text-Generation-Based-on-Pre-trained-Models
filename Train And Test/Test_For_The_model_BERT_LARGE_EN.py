import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

torch.cuda.set_device(5)

# Prediction function
def predict_text(model, tokenizer, text, device='cuda'):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    ai_prob = probabilities[0][1].item()
    return ai_prob

# Command line interface for predictions
def main():
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('/home/group2024-detect4/.vscode-server/data/User/bert-large-uncased')

    # Load model
    model = BertForSequenceClassification.from_pretrained('/home/group2024-detect4/.vscode-server/data/User/bert-large-uncased', num_labels=2)
    model.load_state_dict(torch.load('/home/group2024-detect4/.vscode-server/data/User/Detection_English_BERT_LARGE.pth'))  # Load the trained model parameters
    model = model.to('cuda')

    while True:
        input_text = input("请输入文本进行检测，或输入'exit'退出：")
        if input_text.lower() == 'exit':
            break
        ai_probability = predict_text(model, tokenizer, input_text)
        print(f"AI生成文本的概率是：{ai_probability:.2%}")

if __name__ == "__main__":
    main()
