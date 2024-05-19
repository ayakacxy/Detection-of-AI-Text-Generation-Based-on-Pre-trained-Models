# 项目名称
基于预训练模型的AI生成文本检测
## 项目简介
这个项目旨在基于预训练模型检测AI生成的文本。通过使用各种预训练模型，我们可以有效地识别出由AI生成的文本和人类编写的文本。

预训练模型
BERT: 支持中文和英文文本识别。
RoBERTa: 支持中文和英文文本识别。
GPT-2: 仅支持英文文本识别。

## 安装
请确保你有Python 3环境，并安装所需的依赖包：
可以直接
```bash
pip install Flask Flask-Cors pymupdf torch transformers
```
```bash
pip install -r requirements.txt
```

## 使用方法
1. **下载预训练模型**：
   请从以下链接下载三个预训练的开源模型：
   [预训练模型下载链接](https://rec.ustc.edu.cn/share/436a8ce0-15db-11ef-b9aa-391d78e4e304)。

2. **运行训练脚本**：
   使用下载的预训练模型运行训练脚本，生成自己的检测模型。训练脚本位于项目根目录`Train-And-Test``文件夹下下，以 `AI_detection_` 开头的Python文件。

3. **在 `main.py` 中导入训练好的模型**：
   将训练好的检测模型导入 `main.py` 中，用于文本检测任务。

4. **运行前端和后端**：
   - 打开 `index.html` 作为前端页面。
   - 运行 `main.py` 作为后端服务。
   ```bash
   python main.py

## 贡献
欢迎提交问题和贡献代码。请遵循项目的贡献指南。

## 许可证
本项目使用MIT许可证，详见`LICENSE`文件。

---

你可以根据你的项目具体情况进行调整和补充。
