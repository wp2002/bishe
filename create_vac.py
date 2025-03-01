# 程序功能：
# 本程序主要用于处理存储在 "cleanData" 文件夹中的 JSON 数据文件。
# 它会通过分词（使用 jieba 分词库 或者 直接分词），生成一个词汇表，映射每个词到对应的索引，并将每个文章的内容转换为对应的编码。
# 程序会先从指定数量的文件中提取所有的分词结果，构建词汇表（最大大小 100,000），然后保存词汇表，
# 并将编码后的数据分批保存到 "preData" 文件夹中。
import jieba
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

## 利用cleanData的数据生成词汇表以及相应的映射列表

# 创建 preData 文件夹，如果不存在
os.makedirs('preData', exist_ok=True)

# 是否jieba分词
if_jieba = False

def build_vocab(all_tokens):
    """
    根据所有分词结果构建词汇表，同时生成单词到索引和索引到单词的映射
    :param all_tokens: 所有分词结果的列表
    :return: 词汇表列表、单词到索引的字典、索引到单词的字典
    """
    word_counts = Counter(all_tokens)
    
    # 特殊标记先加入词汇表，包括<END>
    vocab = ['<PAD>', '<UNK>'] + [word for word, count in word_counts.most_common() if word not in ['<PAD>', '<UNK>']]
    
    word_to_index = {word: index for index, word in enumerate(vocab)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    
    return vocab, word_to_index, index_to_word

def encode_text(tokens, word_to_index):
    """
    将分词后的列表转换为对应的索引列表
    :param tokens: 分词后的列表
    :param word_to_index: 单词到索引的字典
    :return: 编码后的索引列表
    """
    encoded_tokens = []
    for token in tokens:
        # 如果 token 不在词汇表中，使用 <UNK> 的索引
        encoded_tokens.append(word_to_index.get(token, word_to_index['<UNK>']))
    return encoded_tokens

def process_line(line):
    """
    处理单行数据，返回分词后的内容
    :param line: 单行文章
    :return: 分词后的结果
    """
    # 使用jieba进行分词
    if if_jieba:
        content_tokens = jieba.lcut(line)
    else:
        content_tokens = list(line)
    return content_tokens

def process_file(file_path):
    """
    处理一个文件，返回所有分词结果
    :param file_path: 文件路径
    :return: 该文件的分词结果
    """
    all_tokens = []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for line in data:
            if line:
                tokens = process_line(line)  # 对每篇文章进行分词
                all_tokens.extend(tokens)
    return all_tokens

def process_files_in_directory(directory_path, num_files=None):
    """
    处理cleanData目录下的所有文件，返回所有文件的分词结果
    :param directory_path: 文件夹路径
    :param num_files: 需要读取的文件数量，如果为 None，读取所有文件
    :return: 所有文件的分词结果
    """
    files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
    
    # 如果指定读取数量，限制读取文件数量
    if num_files:
        files = files[:num_files]

    all_tokens = []
    for filename in files:
        file_path = os.path.join(directory_path, filename)
        print(f"Processing file: {filename}")
        tokens = process_file(file_path)
        all_tokens.extend(tokens)
        print(f"Finished processing file: {filename}")
    
    return all_tokens


def save_vocab(vocab, word_to_index, index_to_word):
    """
    存储词汇表到文件
    :param vocab: 词汇表列表
    :param word_to_index: 单词到索引的字典
    :param index_to_word: 索引到单词的字典
    """
    with open('preData/vocab.json', 'w', encoding='utf-8') as vocab_file:
        json.dump({
            "vocab": vocab,
            "word_to_index": word_to_index,
            "index_to_word": index_to_word
        }, vocab_file, ensure_ascii=False, indent=4)


def save_encoded_data(current_batch, file_count):
    """
    将编码后的数据保存到文件
    :param current_batch: 当前批次的编码数据
    :param file_count: 文件计数
    """
    with open(f'preData/encoded_data_{file_count}.json', 'w', encoding='utf-8') as encoded_file:
        json.dump(current_batch, encoded_file, ensure_ascii=False, indent=4)


def main():
    try:
        # 设置要处理的文件数量
        num_files_to_read = None  # 设置为 None 以读取所有文件，或者设置具体数字读取指定数量的文件

        # 第一步：用前 num_files_to_read 个文件来构建词汇表
        all_tokens_for_vocab = process_files_in_directory('finalData', num_files=num_files_to_read)
        vocab, word_to_index, index_to_word = build_vocab(all_tokens_for_vocab)

        # 保存词汇表
        save_vocab(vocab, word_to_index, index_to_word)
        print("词汇表构建完成。")

        # 第二步：用相同数量的文件来进行编码并保存
        line_count = 0
        file_count = 1
        current_batch = []

        files_processed = 0  # 记录处理的文件数量

        for filename in os.listdir('finalData'):
            if filename.endswith(".json"):
                # 如果 num_files_to_read 为 None，则不限制文件数量
                if num_files_to_read is None or files_processed < num_files_to_read:
                    file_path = os.path.join('finalData', filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for result in data:
                            if result:
                                encoded_content = encode_text(process_line(result), word_to_index)  # 对处理后的内容进行编码
                                if encode_text:
                                    current_batch.append(encoded_content)
                                line_count += 1

                                # 每10000条数据生成一个新的JSON文件
                                if len(current_batch) >= 10000:
                                    save_encoded_data(current_batch, file_count)
                                    print(f"已保存第 {file_count} 个文件，包含 {len(current_batch)} 条数据")
                                    file_count += 1
                                    current_batch = []  # 清空当前批次

                    # 更新处理的文件数量
                    files_processed += 1

                    # 如果已经处理完所有指定的文件，则退出
                    if num_files_to_read is not None and files_processed >= num_files_to_read:
                        break

        # 保存最后一个文件（如果有剩余的数据）
        if current_batch:
            save_encoded_data(current_batch, file_count)
            print(f"已保存第 {file_count} 个文件，包含 {len(current_batch)} 条数据")

        print("数据处理完成，词汇表和编码后的数据已保存到 preData 文件夹中。")

    except FileNotFoundError:
        print("未找到 cleanData 文件夹，请检查文件路径。")
    except json.JSONDecodeError:
        print("无法解析文件中的 JSON 数据，请检查文件格式。")


if __name__ == "__main__":
    main()
