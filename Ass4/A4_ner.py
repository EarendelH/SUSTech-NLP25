import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# 导入评估指标处理类
from metrics import MetricsHandler

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据路径
DATA_DIR = '/home/stu_12310401/nlp/SUSTech-NLP25/Ass4/data'
GLOVE_PATH = '/home/stu_12310401/nlp/SUSTech-NLP25/Ass4/glove.6B.100d.txt'

# 超参数
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# 数据预处理
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word_to_idx, tag_to_idx):
        self.sentences = sentences
        self.tags = tags
        self.word_to_idx = word_to_idx
        self.tag_to_idx = tag_to_idx
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        words = self.sentences[idx]
        tags = self.tags[idx]
        
        # 将单词和标签转换为索引
        word_idxs = [self.word_to_idx.get(word.lower(), self.word_to_idx['<UNK>']) for word in words]
        tag_idxs = [self.tag_to_idx[tag] for tag in tags]
        
        return torch.tensor(word_idxs), torch.tensor(tag_idxs)

def load_data(file_path):
    """加载CoNLL2003格式的NER数据"""
    sentences = []
    tags = []
    
    sentence = []
    sentence_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '' or line.startswith('-DOCSTART-'):
                if sentence:
                    sentences.append(sentence)
                    tags.append(sentence_tags)
                    sentence = []
                    sentence_tags = []
            else:
                parts = line.split()
                if len(parts) >= 4:  # CoNLL格式：单词 POS NP 标签
                    word = parts[0]
                    tag = parts[3]
                    sentence.append(word)
                    sentence_tags.append(tag)
    
    # 添加最后一个句子
    if sentence:
        sentences.append(sentence)
        tags.append(sentence_tags)
    
    return sentences, tags

def build_vocab(sentences, tags, min_freq=1):
    """构建单词和标签的词汇表"""
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update([word.lower() for word in sentence])
    
    # 过滤低频词
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.items():
        if count >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    
    # 构建标签词汇表
    tag_counts = Counter()
    for sentence_tags in tags:
        tag_counts.update(sentence_tags)
    
    tag_to_idx = {'<PAD>': 0}
    for tag in tag_counts:
        tag_to_idx[tag] = len(tag_to_idx)
    
    return word_to_idx, tag_to_idx

def load_glove_embeddings(glove_path, word_to_idx, embedding_dim=100):
    """加载预训练的GloVe词向量"""
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_to_idx), embedding_dim))
    # 将PAD向量设为0
    embeddings[0] = np.zeros(embedding_dim)
    
    # 加载GloVe词向量
    word_count = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="加载GloVe词向量"):
            values = line.split()
            word = values[0]
            if word.lower() in word_to_idx:
                vector = np.array(values[1:], dtype='float32')
                embeddings[word_to_idx[word.lower()]] = vector
                word_count += 1
    
    print(f"加载了 {word_count}/{len(word_to_idx)} 个词的预训练词向量")
    return torch.FloatTensor(embeddings)

def collate_fn(batch):
    """处理不同长度的序列"""
    # 排序批次，按句子长度降序排列
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sentences, tags = zip(*batch)
    
    # 获取每个句子的长度
    lengths = [len(s) for s in sentences]
    max_len = max(lengths)
    
    # 填充序列
    padded_sentences = torch.zeros(len(sentences), max_len).long()
    padded_tags = torch.zeros(len(sentences), max_len).long()
    
    # 填充
    for i, (sentence, tag) in enumerate(zip(sentences, tags)):
        end = lengths[i]
        padded_sentences[i, :end] = sentence[:end]
        padded_tags[i, :end] = tag[:end]
    
    return padded_sentences, padded_tags, torch.tensor(lengths)

# 模型定义
class LSTM_NER(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim, num_layers, dropout, pretrained_embeddings=None):
        super(LSTM_NER, self).__init__()
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)
        
        # 双向LSTM
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim // 2,  # 因为是双向的，所以隐藏维度减半
                           num_layers=num_layers, 
                           bidirectional=True,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        
        # 分类器
        self.fc = nn.Linear(hidden_dim, tag_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths):
        # 获取嵌入
        embedded = self.dropout(self.embedding(x))
        
        # 打包填充序列
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        
        # 通过LSTM
        outputs, _ = self.lstm(packed)
        
        # 解包序列
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        # 应用dropout
        outputs = self.dropout(outputs)
        
        # 通过全连接层
        logits = self.fc(outputs)
        
        return logits

# 训练函数
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for sentences, tags, lengths in tqdm(train_loader, desc="训练"):
        sentences = sentences.to(device)
        tags = tags.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        logits = model(sentences, lengths)
        
        # 计算损失（忽略填充标记）
        loss = 0
        for i in range(logits.size(0)):
            loss += criterion(logits[i, :lengths[i]], tags[i, :lengths[i]])
        loss /= logits.size(0)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# 评估函数
def evaluate(model, data_loader, tag_to_idx, idx_to_tag, metrics_handler, device):
    model.eval()
    
    # Reset metrics handler for new evaluation
    metrics_handler = MetricsHandler(classes=list(tag_to_idx.keys()))
    
    with torch.no_grad():
        for sentences, tags, lengths in tqdm(data_loader, desc="评估"):
            sentences = sentences.to(device)
            tags = tags.to(device)
            
            # 前向传播
            logits = model(sentences, lengths)
            
            # 获取预测结果
            for i in range(logits.size(0)):
                length = lengths[i]
                logits_i = logits[i, :length]
                tags_i = tags[i, :length]
                
                # 贪心解码
                _, predicted = torch.max(logits_i, dim=1)
                
                # 转换为标签
                pred_tags = [idx_to_tag[idx.item()] for idx in predicted]
                true_tags = [idx_to_tag[idx.item()] for idx in tags_i]
                
                # Update metrics handler with this batch's predictions
                metrics_handler.update(pred_tags, true_tags)
    
    # Collect metrics after all batches
    metrics_handler.collect()
    metrics = metrics_handler.get_metrics()
    
    # Calculate the latest F1 score
    f1_scores = metrics["F1-score"]
    latest_f1 = f1_scores[-1] if f1_scores else 0.0
    
    return {"f1": latest_f1, "metrics": metrics}

def main():
    # 加载数据
    print("加载数据...")
    train_sentences, train_tags = load_data(os.path.join(DATA_DIR, 'train.txt'))
    dev_sentences, dev_tags = load_data(os.path.join(DATA_DIR, 'dev.txt'))
    test_sentences, test_tags = load_data(os.path.join(DATA_DIR, 'test.txt'))
    
    # 构建词汇表
    print("构建词汇表...")
    word_to_idx, tag_to_idx = build_vocab(train_sentences, train_tags)
    idx_to_tag = {idx: tag for tag, idx in tag_to_idx.items()}

    TAGSET_SIZE = len(tag_to_idx)
    
    # 加载GloVe词向量
    print("加载GloVe词向量...")
    pretrained_embeddings = load_glove_embeddings(GLOVE_PATH, word_to_idx, EMBEDDING_DIM)
    
    # 创建数据集和数据加载器
    print("创建数据加载器...")
    train_dataset = NERDataset(train_sentences, train_tags, word_to_idx, tag_to_idx)
    dev_dataset = NERDataset(dev_sentences, dev_tags, word_to_idx, tag_to_idx)
    test_dataset = NERDataset(test_sentences, test_tags, word_to_idx, tag_to_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 初始化模型
    print("初始化模型...")
    model = LSTM_NER(
        vocab_size=len(word_to_idx),
        tag_size=len(tag_to_idx),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=tag_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 初始化评估指标处理器
    metrics_handler = MetricsHandler(classes=list(range(TAGSET_SIZE)))
    
    # 训练模型
    print("开始训练...")
    train_losses = []
    dev_f1_scores = []
    
    for epoch in range(NUM_EPOCHS):
        # 训练
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        
        # 在开发集上评估
        dev_metrics = evaluate(model, dev_loader, tag_to_idx, idx_to_tag, metrics_handler, device)
        dev_f1 = dev_metrics['f1']
        dev_f1_scores.append(dev_f1)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, 训练损失: {train_loss:.4f}, 开发集F1: {dev_f1:.4f}")
        
        # 保存前5个epoch的开发集F1分数
        if epoch < 5:
            print(f"Epoch {epoch+1} 开发集F1分数: {dev_f1:.4f}")
    
    # 在测试集上评估
    print("在测试集上评估...")
    test_metrics = evaluate(model, test_loader, tag_to_idx, idx_to_tag, metrics_handler, device)
    test_f1 = test_metrics['f1']
    print(f"测试集F1分数: {test_f1:.4f}")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'word_to_idx': word_to_idx,
        'tag_to_idx': tag_to_idx,
        'idx_to_tag': idx_to_tag,
        'hyperparams': {
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        }
    }, 'lstm_ner_model.pt')
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(dev_f1_scores)
    plt.title('开发集F1分数')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    main()