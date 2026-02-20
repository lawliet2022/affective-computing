import pandas as pd
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

class DeepSeekAPIClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def is_scenario_sentence(self, sentence, max_retries=3):
        """判断句子是否描述了一个情境"""
        
        prompt = f"""
        请分析以下句子是否描述了一个具体的情境或场景：
        
        句子: "{sentence}"
        
        请根据以下标准判断：
        1. 是否描述了一个具体的事件、场景或情况？
        2. 是否包含时间、地点、人物、动作等情境元素？
        3. 是否在讲述一个具体发生的事情而非单纯表达情绪？
        
        如果这是一个描述具体情境的句子，请回答"是"，否则回答"否"。
        只回答"是"或"否"，不要添加其他内容。
        """
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result['choices'][0]['message']['content'].strip().lower()
                    
                    if "是" in answer or "yes" in answer:
                        return True, sentence
                    else:
                        return False, sentence
                else:
                    print(f"API错误 (状态码 {response.status_code}): {response.text}")
                    break
                    
            except Exception as e:
                print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2) 
        
        return False, sentence  

def process_csv_with_api(input_file, output_file, api_key, max_rows=800, batch_size=10):
    """
    处理CSV文件，使用DeepSeek API识别情境句子
    
    参数:
    input_file: 输入CSV文件路径
    output_file: 输出CSV文件路径
    api_key: DeepSeek API密钥
    max_rows: 最大处理行数
    batch_size: 批量处理大小
    """
    
    print(f"正在读取文件: {input_file}")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    total_rows = min(len(df), max_rows)
    print(f"将处理前 {total_rows} 行数据")
    
    client = DeepSeekAPIClient(api_key)
    
    scenario_sentences = []
    processed_count = 0
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        for i in range(total_rows):
            if i >= len(df):
                break
                
            sentence = df.iloc[i]['Sentence']
            label = df.iloc[i]['Label']
            
            future = executor.submit(
                client.is_scenario_sentence,
                sentence
            )
            futures.append((future, sentence, label, i))
        
        for future, sentence, label, idx in futures:
            try:
                is_scenario, processed_sentence = future.result(timeout=35)
                
                if is_scenario:
                    scenario_sentences.append({
                        'Sentence': sentence,
                        'Label': label,
                        'Original_Index': idx
                    })
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"已处理: {processed_count}/{total_rows} 行，找到 {len(scenario_sentences)} 个情境句子")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"处理句子时出错 (第{idx+1}行): {e}")
                processed_count += 1
    
    if scenario_sentences:
        print(f"\n找到 {len(scenario_sentences)} 个描述情境的句子")
        
        result_df = pd.DataFrame(scenario_sentences)
        
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"结果已保存到: {output_file}")
        
        print("\n前5个情境句子示例:")
        for i, row in enumerate(result_df.head(5).itertuples(), 1):
            print(f"{i}. {row.Sentence[:100]}...")
    else:
        print("未找到描述情境的句子")

def simple_rule_based_filter(input_file, output_file, max_rows=800):
    """
    如果API调用有问题，可以使用基于规则的简单过滤作为备选方案
    """
    print("使用基于规则的简单过滤...")
    
    try:
        df = pd.read_csv(input_file)
        total_rows = min(len(df), max_rows)
        
        scenario_sentences = []
        
        scenario_keywords = ['when', 'while', 'after', 'before', 'during', 'because', 
                           'if', 'then', 'was doing', 'were doing', 'saw', 'heard',
                           'went to', 'came to', 'told me', 'said that', 'happened']
        
        for i in range(total_rows):
            if i >= len(df):
                break
                
            sentence = str(df.iloc[i]['Sentence']).lower()
            
            has_scenario_keyword = any(keyword in sentence for keyword in scenario_keywords)
            
            is_long_sentence = len(sentence.split()) > 8
            
            has_action = any(word in sentence for word in ['said', 'went', 'came', 'did', 'made', 'took'])
            
            if (has_scenario_keyword or (is_long_sentence and has_action)):
                scenario_sentences.append({
                    'Sentence': df.iloc[i]['Sentence'],
                    'Label': df.iloc[i]['Label'],
                    'Original_Index': i
                })
        
        if scenario_sentences:
            result_df = pd.DataFrame(scenario_sentences)
            result_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"找到 {len(scenario_sentences)} 个可能的情境句子，已保存到 {output_file}")
        else:
            print("未找到符合规则的句子")
            
    except Exception as e:
        print(f"处理文件时出错: {e}")

def main():
    INPUT_FILE = "anger_data.csv"
    OUTPUT_FILE = "scenario_sentences.csv"
    API_KEY = "sk-4c4fa9877a8c45bbbc08474acafe3389"
    MAX_ROWS = 800
    
    print("=" * 60)
    print("DeepSeek API情境句子分离程序")
    print("=" * 60)
    
    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"成功读取文件: {INPUT_FILE}")
        print(f"文件总行数: {len(df)}")
        print(f"前3行数据预览:")
        print(df.head(3))
        print("-" * 60)
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{INPUT_FILE}'")
        print("请确保文件在当前目录下，或提供正确的文件路径")
        return
    
    # 询问使用哪种方法
    print("请选择处理方法:")
    print("1. 使用DeepSeek API（准确但较慢）")
    print("2. 使用基于规则的简单过滤（快速但可能不准确）")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        print("\n开始使用DeepSeek API处理...")
        print("注意: 处理800行数据可能需要一些时间，请耐心等待")
        process_csv_with_api(INPUT_FILE, OUTPUT_FILE, API_KEY, MAX_ROWS)
    elif choice == "2":
        print("\n开始使用基于规则的过滤...")
        simple_rule_based_filter(INPUT_FILE, OUTPUT_FILE, MAX_ROWS)
    else:
        print("无效选择，默认使用基于规则的过滤")
        simple_rule_based_filter(INPUT_FILE, OUTPUT_FILE, MAX_ROWS)
    
    print("\n程序执行完成!")

if __name__ == "__main__":
    main()