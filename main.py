import torch
from model import LSTMModel
import json
import numpy as np

import warnings
warnings.filterwarnings("ignore")

# 加载模型
def load_model(filename="epoch_2.pth", vocab_size=20000, embed_size=128, hidden_size=128, num_layers=2):
    model = LSTMModel(vocab_size, embed_size, hidden_size, num_layers)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 设定为评估模式
    return model

# 温度采样与top-k采样
def top_k_sampling(logits, k, temperature=1.0):
    # 通过温度调整 logits
    logits = logits / temperature
    # 获取 top-k 的概率分布
    top_k_values, top_k_indices = torch.topk(logits, k)
    top_k_probs = torch.softmax(top_k_values, dim=-1)
    
    # 从 top-k 中进行采样
    sampled_index = torch.multinomial(top_k_probs, 1).item()
    return top_k_indices[0, sampled_index].item()

def generate_text(model, vocab, start_text, max_length=200, top_k=5, temperature=0.5):
    # 将 start_text 转换为词索引，并确保只有在词汇表中存在的词才会被添加
    input_sequence = [vocab.get(word, vocab.get('<UNK>', 1)) for word in list(start_text)]
    print(f"seq = {input_sequence}")
    
    # 如果 start_text 为空或者词汇表中没有匹配的词，则报错或使用默认的初始词汇
    if len(input_sequence) == 0:
        print("Start text contains no valid words from the vocabulary.")
        return ""
    
    # 转换为 LongTensor 并添加 batch_size 维度223
    input_tensor = torch.tensor(input_sequence).unsqueeze(0).to(next(model.parameters()).device).long()

    hidden = None
    output_text = start_text

    with torch.no_grad():
        for _ in range(max_length):
            output, hidden = model(input_tensor, hidden)  # 获取模型输出
            logits = output  # output 是 (batch_size, vocab_size)
            
            # 使用 top-k 和温度采样
            predicted_word_index = top_k_sampling(logits, k=top_k, temperature=temperature)

            # 查找词汇表中的对应词
            predicted_word = next((word for word, idx in vocab.items() if idx == predicted_word_index), None)
            # print(f"predicted_word  = {predicted_word}")
            
            if predicted_word:
                output_text +=  predicted_word  # 将预测的词添加到输出文本
                # 更新输入序列，只保留最新预测的词
                input_tensor = torch.tensor([[predicted_word_index]]).to(input_tensor.device).long()
            else:
                break  # 如果没有找到对应的词，停止生成
            
    return output_text




def main():
    # 加载词汇表
    with open('preData/vocab.json', 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    vocab = vocab_data['word_to_index']

    # 加载模型
    filename = "epoch_0020.pth"
    model = load_model(filename=filename, vocab_size=len(vocab), embed_size=256, hidden_size=256, num_layers=2)
    
    # 设置开始文本
    start_text = "夏天趣事"
    
    # 生成文本
    generated_text = generate_text(model, vocab, start_text, top_k=3, temperature=0.8)
    
    print(f"生成的文本：{generated_text.strip()}")

if __name__ == '__main__':
    main()
