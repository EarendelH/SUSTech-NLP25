import torch.nn as nn
import torch

class BaseModel(nn.Module):
    def __init__(self, word_vocab_size, output_size):
        super(BaseModel, self).__init__()
        ### START YOUR CODE ###
        # 定义模型参数
        self.embedding_dim = 100
        self.hidden_dim = 512
        
        # 词嵌入层
        self.word_embeddings = nn.Embedding(word_vocab_size, self.embedding_dim)
        
        self.sequential1 = nn.Sequential(
            nn.Linear(self.embedding_dim * 6, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.sequential2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )
        self.sequential3 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        
        
        # 输出层
        self.output = nn.Linear(self.hidden_dim//2, output_size)
        
        # 激活函数
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
        self.log_softmax = nn.LogSoftmax(dim=1)
        ### END YOUR CODE ###
    
    def forward(self, x):
        ### START YOUR CODE ###
        # x形状为[batch_size, 6]，代表6个单词的ID
        # 将单词ID转换为嵌入表示
        word_embeds = self.word_embeddings(x.long())  # [batch_size, 6, embedding_dim]
        
        # 将嵌入向量展平
        embeds = word_embeds.view(word_embeds.shape[0], -1)  # [batch_size, 6 * embedding_dim]
        
        # 通过隐藏层
        hidden1 = self.sequential1(embeds)
        hidden2 = self.sequential2(hidden1)
        hidden3 = self.sequential3(hidden2)
        
        # 输出层
        output = self.output(hidden3)
        output = self.log_softmax(output)  
        ### END YOUR CODE ###
        return output


class WordPOSModel(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, output_size):
        super(WordPOSModel, self).__init__()
        ### START YOUR CODE ###
        # 定义模型参数
        self.word_embedding_dim = 50
        self.pos_embedding_dim = 50
        self.hidden_dim = 200
        
        # 词嵌入层和POS标签嵌入层
        self.word_embeddings = nn.Embedding(word_vocab_size, self.word_embedding_dim)
        self.pos_embeddings = nn.Embedding(pos_vocab_size, self.pos_embedding_dim)
        
        # 计算总的嵌入维度
        total_embed_dim = self.word_embedding_dim * 6 + self.pos_embedding_dim * 6
        
        # 隐藏层
        self.sequential1 = nn.Sequential(
            nn.Linear(total_embed_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        self.sequential2 = nn.Sequential(   
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        # 输出层
        self.output = nn.Linear(self.hidden_dim//2, output_size)
        
        # 激活函数
        self.log_softmax = nn.LogSoftmax(dim=1)
        ### END YOUR CODE ###

    def forward(self, x):
        ### START YOUR CODE ###
        # x形状为[batch_size, 12]
        # 前6个是单词ID，后6个是POS标签ID
        word_ids = x[:, :6].long()
        pos_ids = x[:, 6:].long()
        
        # 获取嵌入表示
        word_embeds = self.word_embeddings(word_ids)  # [batch_size, 6, word_embedding_dim]
        pos_embeds = self.pos_embeddings(pos_ids)    # [batch_size, 6, pos_embedding_dim]
        
        # 将嵌入向量展平
        word_embeds_flat = word_embeds.view(word_embeds.shape[0],6, -1)  # [batch_size, 6 * word_embedding_dim]
        pos_embeds_flat = pos_embeds.view(pos_embeds.shape[0],6, -1)    # [batch_size, 6 * pos_embedding_dim]
        
        # 连接词嵌入和POS嵌入
        combined_embeds = torch.cat((word_embeds_flat, pos_embeds_flat), dim=2)

        combined_embeds = combined_embeds.view(combined_embeds.shape[0], -1)
        
        # 通过隐藏层
        hidden1 = self.sequential1(combined_embeds)
        hidden2 = self.sequential2(hidden1)
        
        # 输出层
        output = self.output(hidden2)
        output = self.log_softmax(output)
        ### END YOUR CODE ###
        return output

class BiLSTMWordPOSModel(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, output_size):
        super(BiLSTMWordPOSModel, self).__init__()
        self.word_embedding_dim = 50
        self.pos_embedding_dim = 50
        self.hidden_dim = 200

        
        self.word_embeddings = nn.Embedding(word_vocab_size, self.word_embedding_dim)
        self.pos_embeddings = nn.Embedding(pos_vocab_size, self.pos_embedding_dim)
        
        self.dropout = nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(
            self.word_embedding_dim + self.pos_embedding_dim, 
            self.hidden_dim, 
            num_layers=4,
            batch_first=True, 
            bidirectional=True,
            dropout=0.2 
        )
        
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim * 2)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim * 4)
        
        self.sequential1 = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2)
        )
        self.sequential2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(p=0.2)
        ) 

        self.output = nn.Linear(self.hidden_dim//2, output_size)
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        word_ids = x[:, :6].long()
        pos_ids = x[:, 6:].long()
        
        # 获取嵌入表示
        word_embeds = self.word_embeddings(word_ids)  # [batch_size, 6, word_embedding_dim]
        pos_embeds = self.pos_embeddings(pos_ids)    # [batch_size, 6, pos_embedding_dim]

        # 将嵌入向量展平
        word_embeds_flat = word_embeds.view(word_embeds.shape[0],6, -1)  # [batch_size, 6 * word_embedding_dim]
        pos_embeds_flat = pos_embeds.view(pos_embeds.shape[0],6, -1)    # [batch_size, 6 * pos_embedding_dim]
        
        # 连接词嵌入和POS嵌入
        combined_embeds = torch.cat((word_embeds_flat, pos_embeds_flat), dim=2)

        # combined_embeds = combined_embeds.view(combined_embeds.shape[0], -1)

        lstm_out, _ = self.lstm(combined_embeds)
        lstm_out = self.layer_norm1(lstm_out)
        
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = lstm_out + attn_out  
        
        stack_top = attn_out[:, 0, :]
        buffer_top = attn_out[:, -1, :]
        combined = torch.cat([stack_top, buffer_top], dim=1)
        combined = self.layer_norm2(combined)

        hidden1 = self.sequential1(combined)
        hidden2 = self.sequential2(hidden1)

        output = self.output(hidden2)
        output = self.log_softmax(output)
        return output