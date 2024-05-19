# 基于预训练模型的AI生成文本检测

本项目旨在使用预训练模型检测AI生成的文本。通过利用各种预训练模型，我们可以有效地区分AI生成的文本和人类编写的文本。

## 预训练模型
- **BERT**: 支持中文和英文文本识别。
- **RoBERTa**: 支持中文和英文文本识别。
- **GPT-2**: 仅支持英文文本识别。

## 安装
请确保你有Python 3环境，并安装所需的依赖包：
```bash
pip install Flask Flask-Cors pymupdf torch transformers
```
或者
```bash
pip install -r requirements.txt
```

## 使用方法
1. **下载预训练模型**：
   请从以下链接下载三个预训练的开源模型：
   [预训练模型下载链接](https://rec.ustc.edu.cn/share/436a8ce0-15db-11ef-b9aa-391d78e4e304)。

2. **运行训练脚本**：
   使用下载的预训练模型运行训练脚本，生成自己的检测模型。训练脚本位于项目根目录`Train-And-Test`文件夹下，以 `AI_detection_` 开头的Python文件。

3. **在 `main.py` 中导入训练好的模型**：
   将训练好的检测模型导入 `main.py` 中，用于文本检测任务。

4. **运行前端和后端**：
   - 打开 `index.html` 作为前端页面。
   - 运行 `main.py` 作为后端服务。
   ```bash
   python main.py
   ```
检测文本格式支持``.pdf``和``.txt``最后效果如图所示
![运行结果](结果.png)

## 贡献
我们欢迎任何形式的贡献，请通过提交问题（issue）或拉取请求（pull request）的方式参与。

## 许可证
本项目使用MIT许可证，详见`LICENSE`文件。

## 作者
- 俞樾奕
- 谢旋超
- 陈夏雨
- 张如顺
- 廖健铭

