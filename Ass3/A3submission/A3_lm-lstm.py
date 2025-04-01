# LSTM on Harry_Potter_all_books_preprocessed.txt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import re
import random
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from utils import build_training_visualization

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Device configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 只使用GPU 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT_RATE = 0.2
BATCH_SIZE = 64
SEQ_LENGTH = 50
LEARNING_RATE = 0.001
NUM_EPOCHS = 10  
MIN_WORD_FREQ = 2

# File paths
DATA_PATH = '/home/stu_12310401/nlp/SUSTech-NLP25/Ass3/Harry_Potter_all_books_preprocessed.txt'
MODEL_SAVE_PATH = '/home/stu_12310401/nlp/SUSTech-NLP25/Ass3/lstm_lm_model-random200.pth'
VISUALIZATION_PATH = '/home/stu_12310401/nlp/SUSTech-NLP25/Ass3/lstm_lm_training-random200.png'

# 下载NLTK的tokenizer数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_and_preprocess_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 使用NLTK进行单词级别的tokenization
    tokens = word_tokenize(text)
    
    # 构建词汇表（包含特殊标记）
    word_counts = Counter(tokens)
    vocab = ['<PAD>', '<UNK>', '<START>', '<END>'] + [word for word, count in word_counts.most_common() if count >= MIN_WORD_FREQ]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for idx, word in enumerate(vocab)}
    
    # 将文本转换为索引序列
    text_indices = []
    for token in tokens:
        if token in word_to_idx:
            text_indices.append(word_to_idx[token])
        else:
            text_indices.append(word_to_idx['<UNK>'])
    
    return ' '.join(tokens), text_indices, word_to_idx, idx_to_word, len(vocab)

# Dataset class
class TextDataset(Dataset):
    def __init__(self, text_indices, seq_length):
        self.text_indices = text_indices
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.text_indices) - self.seq_length
    
    def __getitem__(self, idx):
        # Get sequence and target
        sequence = self.text_indices[idx:idx+self.seq_length]
        target = self.text_indices[idx+1:idx+self.seq_length+1]
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(target, dtype=torch.long)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout_rate if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
    def forward(self, x, hidden=None):
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
            
        # Embedding layer
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM layer
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        
        # Fully connected layer
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))

# Training function
def train(model, dataloader, criterion, optimizer, clip_value=5.0):
    model.train()
    total_loss = 0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Initialize hidden state
        hidden = model.init_hidden(inputs.size(0))
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs, hidden = model(inputs, hidden)
        
        # Reshape outputs and targets for loss calculation
        outputs = outputs.reshape(-1, outputs.size(2))
        targets = targets.reshape(-1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Evaluation function
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Initialize hidden state
            hidden = model.init_hidden(inputs.size(0))
            
            # Forward pass
            outputs, hidden = model(inputs, hidden)
            
            # Reshape outputs and targets for loss calculation
            outputs = outputs.reshape(-1, outputs.size(2))
            targets = targets.reshape(-1)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

# Generate text function
def generate_text(model, seed_text, word_to_idx, idx_to_word, length=20, temperature=1.0):
    model.eval()
    
    # 对种子文本进行分词
    seed_tokens = word_tokenize(seed_text)
    
    # 转换为索引
    seed_indices = []
    for token in seed_tokens:
        if token in word_to_idx:
            seed_indices.append(word_to_idx[token])
        else:
            seed_indices.append(word_to_idx['<UNK>'])
    
    # 确保序列长度正确
    if len(seed_indices) < SEQ_LENGTH:
        seed_indices = [word_to_idx['<PAD>']] * (SEQ_LENGTH - len(seed_indices)) + seed_indices
    elif len(seed_indices) > SEQ_LENGTH:
        seed_indices = seed_indices[-SEQ_LENGTH:]
    
    current_indices = seed_indices.copy()
    generated_tokens = seed_tokens.copy()
    
    # 初始化隐藏状态
    hidden = model.init_hidden(1)
    
    with torch.no_grad():
        for _ in range(length):
            x = torch.tensor([current_indices], dtype=torch.long).to(device)
            output, hidden = model(x, hidden)
            
            # 获取最后一个时间步的预测
            output = output[0, -1, :] / temperature
            probabilities = torch.softmax(output, dim=0)
            
            # 采样下一个词
            next_index = torch.multinomial(probabilities, 1).item()
            
            # 添加到生成的文本中
            generated_tokens.append(idx_to_word[next_index])
            
            # 更新当前序列
            current_indices = current_indices[1:] + [next_index]
    
    return ' '.join(generated_tokens)

def calculate_perplexity(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_words = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Calculating Perplexity"):
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = model.init_hidden(inputs.size(0))
            outputs, _ = model(inputs, hidden)
            
            outputs = outputs.reshape(-1, outputs.size(2))
            targets = targets.reshape(-1)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_words += targets.size(0)
    
    avg_loss = total_loss / total_words
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

def main():
    print("Loading and preprocessing data...")
    text, text_indices, word_to_idx, idx_to_word, vocab_size = load_and_preprocess_data(DATA_PATH)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset and dataloader
    dataset = TextDataset(text_indices, SEQ_LENGTH)
    
    # Split into train, validation, and test sets (90%, 5%, 5%)
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = int(0.05 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = LSTMModel(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT_RATE).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []  # 修改为验证集损失
    test_losses = []
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train(model, train_dataloader, criterion, optimizer)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss = evaluate(model, val_dataloader, criterion)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        test_loss = evaluate(model, test_dataloader, criterion)
        test_losses.append(test_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Generate sample text
        seed_text = text[:SEQ_LENGTH]
        generated_text = generate_text(model, seed_text, word_to_idx, idx_to_word, length=30)
        print(f"Generated Text Sample:\n{generated_text}\n")
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, MODEL_SAVE_PATH)
    
    # Visualize training progress
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(VISUALIZATION_PATH)
    
    # Create more detailed visualization using the utility function
    train_metrics = {'loss': train_losses}
    validation_metrics = {'loss': val_losses}
    build_training_visualization('LSTM Language Model', train_metrics, train_losses, validation_metrics, VISUALIZATION_PATH)
    
    print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
    print(f"Training visualization saved to {VISUALIZATION_PATH}")
    
    # Calculate final perplexity scores
    print("\nCalculating final perplexity scores...")
    train_perplexity = calculate_perplexity(model, train_dataloader, criterion)
    test_perplexity = calculate_perplexity(model, test_dataloader, criterion)
    
    print(f"\nFinal Perplexity Scores:")
    print(f"Train Perplexity: {train_perplexity:.2f}")
    print(f"Test Perplexity: {test_perplexity:.2f}")

    # Save the results
    results = {
        'train_perplexity': train_perplexity,
        'test_perplexity': test_perplexity,
        'train_losses': train_losses,
        'test_losses': test_losses
    }
    torch.save(results, MODEL_SAVE_PATH.replace('.pth', '_results.pth'))
    
    # Generate final sample
    seed_text = "Harry Potter "
    generated_text = generate_text(model, seed_text, word_to_idx, idx_to_word, length=200, temperature=0.8)
    print(f"Final Generated Text Sample:\n{generated_text}")

if __name__ == "__main__":
    main()