import pandas as pd



# 从 train_data.csv 中分出 500 行作为测试集，并从训练集中去除
train_df = pd.read_csv('train_data.csv')
test_df = train_df.sample(n=500, random_state=42)
train_remain = train_df.drop(test_df.index).reset_index(drop=True)

test_df.to_csv('test_data.csv', index=False, encoding='utf-8')
train_remain.to_csv('train_data.csv', index=False, encoding='utf-8')

print(f"已从 train_data.csv 分出 500 行保存到 test_data.csv")
print(f"train_data.csv 剩余 {len(train_remain)} 行")