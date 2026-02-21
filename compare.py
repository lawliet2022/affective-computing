import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

MAX_LENGTH = 128
TEST_FILE = "data/test_data.csv"
FINETUNED_DIR = "./model/bert-base-uncased-emotion"
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_tokenize(path):
    df = pd.read_csv(path)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].values
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    return enc, torch.tensor(labels, dtype=torch.long), tokenizer

def evaluate(model, enc, labels, tokenizer):
    model.eval()
    model.to(DEVICE)
    n = labels.size(0)
    preds = []
    with torch.no_grad():
        for i in range(0, n, BATCH_SIZE):
            batch = {k: v[i : i + BATCH_SIZE].to(DEVICE) for k, v in enc.items()}
            out = model(**batch)
            preds.append(out.logits.argmax(dim=1).cpu())
    preds = torch.cat(preds, dim=0)
    acc = (preds == labels).float().mean().item()
    return acc

def main():
    print("加载测试集与分词器...")
    enc, labels, tokenizer = load_and_tokenize(TEST_FILE)
    labels = labels  # keep on CPU for comparison

    print("加载原始 BERT 并评估...")
    model_orig = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    acc_orig = evaluate(model_orig, enc, labels, tokenizer)

    print("加载微调后 BERT 并评估...")
    model_ft = BertForSequenceClassification.from_pretrained(FINETUNED_DIR)
    acc_ft = evaluate(model_ft, enc, labels, tokenizer)

    print("\n" + "=" * 50)
    print("在 test_data.csv 上的准确率对比")
    print("=" * 50)
    print(f"  原始 BERT:     {acc_orig:.2%}")
    print(f"  微调后 BERT:   {acc_ft:.2%}")
    print(f"  提升:          {acc_ft - acc_orig:+.2%}")
    print("=" * 50)

if __name__ == "__main__":
    main()
