import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import json

class TextDataset(Dataset):
    """
    文本数据集类，用于加载和预处理文本数据。
    """
    def __init__(self, file_paths):
        """
        初始化函数。
        Args:
            file_paths (list): 包含预处理后JSON文件路径的列表。
        """
        self.data = []  # 用于存储数据的列表

        for file_path in tqdm(file_paths, desc="Processing files", unit="file"):  # 使用tqdm显示进度条
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    content = json.load(f)  # 加载JSON文件内容
                    for encoded_content in content:  # 遍历JSON文件中的每个编码内容
                        if isinstance(encoded_content, list):  # 检查内容是否为列表
                            self.data.append(encoded_content)  # 将编码内容添加到数据列表中
                        else:
                            print(f"Warning: Skipping invalid data in file {file_path} (not a list).")  # 打印警告信息
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON in file {file_path}.")  # 打印JSON解码错误警告
                    continue  # 跳过当前文件

    def __len__(self):
        """
        返回数据集大小。

        Returns:
            int: 数据集大小。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引获取数据。

        Args:
            idx (int): 数据索引。

        Returns:
            list: 对应索引的数据。
        """
        return self.data[idx]

def pad_sequences(sequences, padding_value=0):
    """
    填充序列，使所有序列具有相同的长度。
    Args:
        sequences (list of list): 输入的序列数据
        padding_value (int): 填充使用的值
    Returns:
        torch.Tensor: 填充后的序列张量
    """
    return pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in sequences], batch_first=True, padding_value=padding_value)

def get_data_loader(file_paths, batch_size=32, test_size=0.2):
    """
    获取数据加载器。

    返回：训练集和验证集的 DataLoader，以及每个序列的实际长度
    """
    dataset = TextDataset(file_paths)  # 创建TextDataset实例
    train_data, val_data = train_test_split(dataset.data, test_size=test_size, random_state=42)  # 划分训练集和验证集

    # 填充数据，使所有序列长度一致
    train_data_padded = pad_sequences(train_data)
    val_data_padded = pad_sequences(val_data)

    # 获取每个序列的实际长度
    train_lengths = [len(seq) for seq in train_data]
    val_lengths = [len(seq) for seq in val_data]

    # 将数据转换为 TensorDataset
    train_dataset = torch.utils.data.TensorDataset(train_data_padded, torch.tensor(train_lengths, dtype=torch.long))
    val_dataset = torch.utils.data.TensorDataset(val_data_padded, torch.tensor(val_lengths, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
