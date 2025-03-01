import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
import math
from model import LSTMModel
from data_loader import get_data_loader
from nltk.translate.bleu_score import sentence_bleu
import json
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(filename='train_log.log', level=logging.INFO)

# 计算困惑度
def calculate_perplexity(loss):
    return math.exp(loss)

# calculate_bleu 函数完整实现
def calculate_bleu(predicted_ids, target_ids, vocab):
    """
    计算BLEU-4分数（需要提前安装nltk：pip install nltk）
    :param predicted_ids: 模型输出的token索引列表 (List[int])
    :param target_ids: 真实token索引列表 (List[int])
    :param vocab: 包含index_to_word映射的词典
    :return: BLEU-4分数
    """
    from nltk.translate.bleu_score import sentence_bleu

    # 过滤特殊标记（假设0: <PAD>, 1: <UNK>, 2: <END>）
    predicted_tokens = [
        vocab['index_to_word'].get(int(i), '<UNK>')  # 处理不存在的索引
        for i in predicted_ids 
        if int(i) not in [0, 1, 2]
    ]

    target_tokens = [
        vocab['index_to_word'][i] 
        for i in target_ids 
        if i not in [0, 1, 2]
    ]

    # 至少需要4个词才能计算BLEU-4
    if len(predicted_tokens) < 4 or len(target_tokens) < 4:
        return 0.0

    return sentence_bleu(
        [target_tokens],
        predicted_tokens,
        weights=(0.25, 0.25, 0.25, 0.25)  # BLEU-4
    )




def train(model, train_loader, optimizer, epoch, device, vocab):
    model.train()
    total_loss = 0
    total_bleu = 0
    total_perplexity = 0
    
    # 自动混合精度配置
    use_amp = torch.cuda.is_available()  # 仅在GPU可用时启用
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 梯度累积步数（根据显存调整）
    accumulation_steps = 4 if torch.cuda.is_available() else 1
    accumulation_counter = 0

    # 兼容CPU/GPU的数据预取
    prefetch_factor = 2 if torch.cuda.is_available() else None
    train_loader = torch.utils.data.DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=8,  # 根据机器的CPU核心数来设置
        pin_memory=True,  # 将数据预加载到GPU内存
        prefetch_factor=8  # 预取数据的数量
    )

    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}") as pbar:
        for batch_idx, (data, lengths) in pbar:
            try:
                # 数据转移到设备
                data = data.to(device, non_blocking=True)
                lengths = lengths.to(device, non_blocking=True)

                # 数据有效性过滤
                adjusted_lengths = [max(l-1, 1) for l in lengths.cpu().tolist()]
                valid_indices = [i for i, l in enumerate(adjusted_lengths) if l > 0]
                if not valid_indices:
                    continue
                
                data = data[valid_indices]
                lengths = lengths[valid_indices]
                adjusted_lengths = [l for l in adjusted_lengths if l > 0]

                # 输入/目标处理
                inputs = data[:, :-1]
                targets = data[:, 1:]
                
                # 混合精度前向传播
                with torch.cuda.amp.autocast(enabled=use_amp):
                    embedded = model.embedding(inputs)
                    packed_input = pack_padded_sequence(
                        embedded, adjusted_lengths,
                        batch_first=True, enforce_sorted=False
                    )
                    packed_output, _ = model.lstm(packed_input)
                    output, _ = pad_packed_sequence(packed_output, batch_first=True)
                    output = model.fc(output)
                    
                    # 动态掩码生成
                    seq_len = output.size(1)
                    targets = targets[:, :seq_len]  # 自动对齐长度
                    mask = torch.arange(seq_len, device=device)[None, :] < torch.tensor(adjusted_lengths, device=device)[:, None]
                    
                    # 损失计算
                    masked_output = output[mask]
                    masked_targets = targets[mask]
                    loss = F.cross_entropy(masked_output, masked_targets) / accumulation_steps

                # 混合精度反向传播
                scaler.scale(loss).backward()
                accumulation_counter += 1

                # 梯度累积更新
                if accumulation_counter % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)  # 更高效的显存释放
                    accumulation_counter = 0

                # 统计指标
                with torch.no_grad():
                    total_loss += loss.item() * accumulation_steps
                    predicted_ids = torch.argmax(output, dim=-1)[mask].cpu().numpy().tolist()
                    target_ids = targets[mask].cpu().numpy().tolist()
                    bleu_score = calculate_bleu(predicted_ids, target_ids, vocab)
                    total_bleu += bleu_score
                    total_perplexity += math.exp(loss.item())

                # 动态进度更新
                pbar.set_postfix({
                    'Loss': f"{loss.item()*accumulation_steps:.3f}",
                    'BLEU': f"{bleu_score:.2f}",
                    'Perplexity': f"{math.exp(loss.item()):.1f}"
                })

            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"\nBatch {batch_idx} OOM, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

    # 计算平均指标
    avg_loss = total_loss / len(train_loader)
    avg_bleu = total_bleu / len(train_loader)
    avg_perplexity = total_perplexity / len(train_loader)
    
    logging.info(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}, Avg BLEU: {avg_bleu:.4f}, Avg Perplexity: {avg_perplexity:.4f}")
    print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}, Avg BLEU: {avg_bleu:.4f}, Avg Perplexity: {avg_perplexity:.4f}")

    # 显存清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
def save_model(model, optimizer, epoch, filename="lstm_model.pth"):
    """
    保存模型和优化器的状态字典。

    :param model: 训练好的模型
    :param optimizer: 优化器
    :param epoch: 当前 epoch
    :param filename: 保存文件的名称
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filename)
    logging.info(f"Model saved at epoch {epoch}")

def load_checkpoint(model, optimizer, filename="lstm_model.pth"):
    """
    加载模型和优化器的状态字典。

    :param model: LSTM 模型
    :param optimizer: 优化器
    :param filename: checkpoint 文件
    :return: 加载的 epoch 和模型
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch, model, optimizer

def main():
    """
    主函数，包含模型初始化、数据加载和训练的过程。
    """
    torch.backends.cudnn.benchmark = True   # 自动选择最优卷积算法
    torch.backends.cudnn.enabled = True     # 启用cuDNN加速

    # 超参数
    batch_size = 128
    epochs = 1000
    embed_size = 256
    hidden_size = 256
    num_layers = 2
    learning_rate = 0.01
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载词汇表
    with open('preData/vocab.json', 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    # 转换index_to_word的键为整数（JSON默认保存为字符串）
    vocab = {
        'word_to_index': vocab_data['word_to_index'], 
        'index_to_word': {int(k): v for k, v in vocab_data['index_to_word'].items()}
    }
    
    # 构建文件名列表
    start_index = 1
    end_index = 7  # 读取文件 1 到 7
    file_paths = [f'preData/encoded_data_{i}.json' for i in range(start_index, end_index + 1)]

    # 初始化数据加载器
    train_loader, val_loader = get_data_loader(file_paths, batch_size=batch_size)

    # 初始化模型
    model = LSTMModel(len(vocab['word_to_index']), embed_size, hidden_size, num_layers).to(device)
    print("LSTM模型初始化成功")

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print("optimizer初始化成功")

    # 加载检查点
    start_epoch = 1
    filename = "checkpoint.pth"
    if os.path.exists(filename):
        start_epoch, model, optimizer = load_checkpoint(model, optimizer, filename=filename)
        print(f"Resuming from epoch {start_epoch}")

    # 开始训练
    for epoch in range(start_epoch, epochs + 1):
        # 训练过程
        train(model, train_loader, optimizer, epoch, device, vocab)  # 训练过程中使用 train_loader 和 val_loader

        checkpoint_filename = f"checkpoint.pth"
        save_model(model, optimizer, epoch, filename=checkpoint_filename)

        # 每个epoch结束后对验证集进行评估
        if epoch % 20 == 0:
            val_loss, val_bleu, val_perplexity = evaluate(model, val_loader, device, vocab)
            logging.info(f"Epoch {epoch} Validation - Loss: {val_loss:.4f}, BLEU: {val_bleu:.4f}, Perplexity: {val_perplexity:.4f}")
            print(f"Epoch {epoch} Validation - Loss: {val_loss:.4f}, BLEU: {val_bleu:.4f}, Perplexity: {val_perplexity:.4f}")
            filename = f"epoch_{epoch:04d}.pth"
            save_model(model, optimizer, epoch, filename=filename)

def evaluate(model, val_loader, device, vocab):
    """
    用于评估验证集，计算损失、BLEU分数和困惑度。
    """
    model.eval()
    total_loss = 0
    total_bleu = 0
    total_perplexity = 0
    
    with torch.no_grad():
        for data, lengths in val_loader:
            data = data.to(device)
            lengths = lengths.to(device)

            adjusted_lengths = [max(l - 1, 1) for l in lengths.cpu().tolist()]
            valid_indices = [i for i, l in enumerate(adjusted_lengths) if l > 0]
            if not valid_indices:
                continue

            data = data[valid_indices]
            lengths = lengths[valid_indices]
            adjusted_lengths = [l for l in adjusted_lengths if l > 0]

            inputs = data[:, :-1]
            targets = data[:, 1:]

            embedded = model.embedding(inputs)
            packed_input = pack_padded_sequence(embedded, adjusted_lengths, batch_first=True, enforce_sorted=False)
            packed_output, _ = model.lstm(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            output = model.fc(output)

            seq_len = output.size(1)
            targets = targets[:, :seq_len]  # 自动对齐长度
            mask = torch.arange(seq_len, device=device)[None, :] < torch.tensor(adjusted_lengths, device=device)[:, None]
            
            # 损失计算
            masked_output = output[mask]
            masked_targets = targets[mask]
            loss = F.cross_entropy(masked_output, masked_targets)
            
            total_loss += loss.item()
            predicted_ids = torch.argmax(output, dim=-1)[mask].cpu().numpy().tolist()
            target_ids = targets[mask].cpu().numpy().tolist()
            bleu_score = calculate_bleu(predicted_ids, target_ids, vocab)
            total_bleu += bleu_score
            total_perplexity += math.exp(loss.item())

    # 计算平均指标
    avg_loss = total_loss / len(val_loader)
    avg_bleu = total_bleu / len(val_loader)
    avg_perplexity = total_perplexity / len(val_loader)
    
    return avg_loss, avg_bleu, avg_perplexity

if __name__ == '__main__':
    main()