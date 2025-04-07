from pprint import pprint
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import Indexer, read_ner_data_from_connl, get_batch
from metrics import MetricsHandler

torch.manual_seed(42)
np.random.seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = "4" 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

TRAIN_PATH = 'data/train.txt'
DEV_PATH = 'data/dev.txt'
TEST_PATH = 'data/test.txt'
EMBEDDINGS_PATH = 'glove.6B.100d.txt' 

BATCH_SIZE = 128
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

def prepare_data():
    train_words, train_tags = read_ner_data_from_connl(TRAIN_PATH)
    dev_words, dev_tags = read_ner_data_from_connl(DEV_PATH)
    test_words, test_tags = read_ner_data_from_connl(TEST_PATH)

    indexer_train_words = Indexer(train_words)
    indexer_train_tags = Indexer(train_tags)

    all_words = train_words + dev_words + test_words
    all_tags = train_tags + dev_tags + test_tags
    
    indexer_words = Indexer(all_words)
    indexer_tags = Indexer(all_tags)

    batches = list(get_batch(train_words, train_tags, BATCH_SIZE))

    return indexer_words, indexer_tags, batches, train_words, train_tags, dev_words, dev_tags, test_words, test_tags

def load_glove_embeddings(embeddings_path, indexer_words, embedding_dim=100):
    """加载预训练的GloVe词向量"""
    vocab_size = len(indexer_words)
    embeddings = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embeddings[0] = np.zeros(embedding_dim)
    
    word_count = 0
    with open(embeddings_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            if word.lower() in indexer_words._element_to_index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[indexer_words._element_to_index[word.lower()]] = vector
                word_count += 1
    
    print(f"加载了 {word_count}/{vocab_size} 个词的预训练词向量")
    return torch.FloatTensor(embeddings)


class BiLSTM_NER(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim, hidden_dim, num_layers, dropout, pretrained_embeddings=None):
        super(BiLSTM_NER, self).__init__()  
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings)

        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim // 2,  
                           num_layers=num_layers, 
                           bidirectional=True,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        

        self.fc = nn.Linear(hidden_dim, tag_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, lengths):

        embedded = self.dropout(self.embedding(x))
        
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        outputs, _ = self.lstm(packed)
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        outputs = self.dropout(outputs)
        
        logits = self.fc(outputs)
        
        return logits

def train_epoch(model, batches, optimizer, criterion, indexer_words, indexer_tags, device):
    model.train()
    total_loss = 0
    
    for words_batch, tags_batch in tqdm(batches):

        batch_size = len(words_batch)
        max_len = max(len(words) for words in words_batch)
        
        words_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
        tags_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
        lengths = torch.tensor([len(words) for words in words_batch])
        
        for i, (words, tags) in enumerate(zip(words_batch, tags_batch)):
            for j, word in enumerate(words):

                words_tensor[i, j] = indexer_words.element_to_index(word.lower())
            for j, tag in enumerate(tags):
                tags_tensor[i, j] = indexer_tags.element_to_index(tag)
        

        words_tensor = words_tensor.to(device)
        tags_tensor = tags_tensor.to(device)
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        logits = model(words_tensor, lengths)
        
        loss = 0
        for i in range(batch_size):
            loss += criterion(logits[i, :lengths[i]], tags_tensor[i, :lengths[i]])
        loss /= batch_size
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(batches)

def evaluate(model, words_data, tags_data, indexer_words, indexer_tags, metrics_handler, device, batch_size=32):
    model.eval()
    
    batches = list(get_batch(words_data, tags_data, batch_size))
    
    with torch.no_grad():
        for words_batch, tags_batch in tqdm(batches, desc="评估"):
            batch_size = len(words_batch)
            max_len = max(len(words) for words in words_batch)
            
            words_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
            lengths = torch.tensor([len(words) for words in words_batch])
            
            for i, words in enumerate(words_batch):
                for j, word in enumerate(words):
                    words_tensor[i, j] = indexer_words.element_to_index(word.lower())
            
            words_tensor = words_tensor.to(device)
            lengths = lengths.to(device)
            
            logits = model(words_tensor, lengths)
            
            for i in range(batch_size):
                length = lengths[i].item()  
                logits_i = logits[i, :length]
                
                # 贪心
                _, predicted = torch.max(logits_i, dim=1)
                
                pred_tags = [indexer_tags.index_to_element(idx.item()) for idx in predicted]
                true_tags = tags_batch[i]
                
                min_len = min(len(pred_tags), len(true_tags))
                pred_tags = pred_tags[:min_len]
                true_tags = true_tags[:min_len]
                
                metrics_handler.update(pred_tags, true_tags)
    
    metrics_handler.collect()
    metrics = metrics_handler.get_metrics()
    
    f1_scores = metrics["F1-score"]
    latest_f1 = f1_scores[-1] if f1_scores else 0.0
    
    return {"f1": latest_f1, "metrics": metrics}

def main():
    indexer_words, indexer_tags, batches, train_words, train_tags, dev_words, dev_tags, test_words, test_tags = prepare_data()
    
    pretrained_embeddings = load_glove_embeddings(EMBEDDINGS_PATH, indexer_words, EMBEDDING_DIM)
    
    model = BiLSTM_NER(
        vocab_size=len(indexer_words),
        tag_size=len(indexer_tags),
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pretrained_embeddings=pretrained_embeddings
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    

    unique_tags = list(indexer_tags._element_to_index.keys())
    metrics_handler = MetricsHandler(classes=unique_tags)
    
    print("开始训练...")
    train_losses = []
    dev_f1_scores = []
    
    for epoch in range(NUM_EPOCHS):
  
        train_loss = train_epoch(model, batches, optimizer, criterion, indexer_words, indexer_tags, device)
        train_losses.append(train_loss)
        
        dev_metrics = evaluate(model, dev_words, dev_tags, indexer_words, indexer_tags, metrics_handler, device)
        dev_f1 = dev_metrics['f1']
        dev_f1_scores.append(dev_f1)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, 训练损失: {train_loss:.4f}, 开发集F1: {dev_f1:.4f}")
        
        if epoch < 5:
            print(f"Epoch {epoch+1} 开发集F1分数: {dev_f1:.4f}")
    
    print("在测试集上评估...")
    test_metrics = evaluate(model, test_words, test_tags, indexer_words, indexer_tags, metrics_handler, device)
    test_f1 = test_metrics['f1']
    print(f"测试集F1分数: {test_f1:.4f}")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'indexer_words': indexer_words,
        'indexer_tags': indexer_tags,
        'hyperparams': {
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT
        }
    }, 'bilstm_ner_model.pt')
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(dev_f1_scores)
    plt.title('F1 Score on Dev Set')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

if __name__ == "__main__":
    main()




