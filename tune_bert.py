import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 1. 加载预训练模型和分词器
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. 加载数据集：训练集与验证集分开读取
dataset = load_dataset(
    "csv",
    data_files={
        "train": "data/train_data.csv",
        "test": "data/test_data.csv",
    },
)

# 3. 预处理函数（缩短 max_length 可明显加速，情绪短句 128 通常够用）
MAX_LENGTH = 128

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 将 label 列改名为 labels，供 Trainer 使用
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 评估时计算正确率
def compute_metrics(eval_pred):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=1)
    return {"accuracy": (preds == labels).mean()}

# 4. 设置训练参数（混合精度 + 多线程加载 可加速）
training_args = TrainingArguments(
    output_dir="./model/bert-base-uncased-emotion",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    dataloader_num_workers=0,
)

# 5. 创建 Trainer 并开始训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()

# 6. 评估模型并输出正确率
eval_result = trainer.evaluate()
print("评估结果:", eval_result)
print("正确率 (accuracy): {:.2%}".format(eval_result["eval_accuracy"]))

# 7. 保存模型

trainer.save_model("./model/bert-base-uncased-emotion")
