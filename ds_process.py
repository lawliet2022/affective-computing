import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('emotions_dataset.parquet')

# 筛选 Label 为 'anger' 的行
# 注意：根据实际情况可能需要调整大小写，如 'anger' vs 'Anger' vs 'ANGER'
anger_df = df[df['Label'] == 'anger']

# 或者使用更灵活的方式（忽略大小写）
anger_df = df[df['Label'].str.lower() == 'anger']

# 保存为 CSV 文件
anger_df.to_csv('anger_data.csv', index=False, encoding='utf-8')

print(f"找到 {len(anger_df)} 行 Label 为 anger 的数据")
print(f"已保存到 anger_data.csv")