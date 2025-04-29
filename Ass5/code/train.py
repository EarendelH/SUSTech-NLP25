import sys
import numpy as np
import datetime
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from get_train_data import FeatureExtractor
from model import BaseModel, WordPOSModel, BiLSTMWordPOSModel

argparser = argparse.ArgumentParser()
argparser.add_argument('--input_file', default='input_train.npy')
argparser.add_argument('--target_file', default='target_train.npy')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt')
argparser.add_argument('--model', default='model2.pt', help='path to save model file, if not specified, a .pt with timestamp will be used')
argparser.add_argument('--model_type', choices=['base', 'wordpos','bilstm'], default='base', help='type of model to train')
# 添加GPU相关参数
argparser.add_argument('--gpu_id', type=int, default=2, help='GPU设备ID，默认使用第三张GPU卡 (ID=2)')
argparser.add_argument('--no_cuda', action='store_true', help='禁用CUDA即使可用')

if __name__ == "__main__":
    args = argparser.parse_args()
    
    # 设置GPU设备
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    if use_cuda:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        print(f"使用GPU设备: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print(f"使用CPU设备")
    
    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
        rel_vocab_file = open(args.rel_vocab, "r")
    except FileNotFoundError:
        print(f'Could not find vocabulary files {args.words_vocab}, {args.pos_vocab}, and {args.rel_vocab}')
        sys.exit(1)
    
    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file, rel_vocab_file)
    word_vocab_size = len(extractor.word_vocab)
    pos_vocab_size = len(extractor.pos_vocab)
    output_size = len(extractor.rel_vocab)

    ### START YOUR CODE ###
    # 根据选择的模型类型初始化模型
    print(f"初始化{args.model_type}模型...")
    if args.model_type == 'base':
        model = BaseModel(word_vocab_size, output_size)
    elif args.model_type == 'wordpos':
        model = WordPOSModel(word_vocab_size, pos_vocab_size, output_size)
    elif args.model_type == 'bilstm':
        model = BiLSTMWordPOSModel(word_vocab_size, pos_vocab_size, output_size)
    else:
        raise ValueError(f"不支持的模型类型: {args.model_type}")
    
    # 将模型移动到GPU
    model = model.to(device)
    ### END YOUR CODE ###

    # 使用更合适的优化器参数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.NLLLoss().to(device)

    # 加载数据
    inputs = np.load(args.input_file)
    targets = np.load(args.target_file)
    print(f"加载数据完成。输入形状: {inputs.shape}, 目标形状: {targets.shape}")
    
    # 验证输入维度与模型匹配
    input_dim = inputs.shape[1]
    expected_dim = 6 if args.model_type == 'base' else 12
    if input_dim != expected_dim:
        print(f"警告: 输入数据维度 ({input_dim}) 与所选模型 ({args.model_type}, 期望维度 {expected_dim}) 不匹配!")
        sys.exit(1)

    # 划分训练集和验证集
    num_samples = len(inputs)
    indices = np.random.permutation(num_samples)
    val_size = int(0.1 * num_samples)  # 10%作为验证集
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_inputs = inputs[train_indices]
    train_targets = targets[train_indices]
    val_inputs = inputs[val_indices]
    val_targets = targets[val_indices]

    # Train loop
    n_epochs = 5
    batch_size = 2048  # 使用GPU后可以增大批量大小
    print_loss_every = 100
    patience = 3  # 早停参数
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    ### START YOUR CODE ###
    # 转换为PyTorch张量并移动到GPU
    train_inputs_tensor = torch.tensor(train_inputs, dtype=torch.float32).to(device)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.long).to(device)
    val_inputs_tensor = torch.tensor(val_inputs, dtype=torch.float32).to(device)
    val_targets_tensor = torch.tensor(val_targets, dtype=torch.long).to(device)
    ### END YOUR CODE ###

    # 创建数据加载器
    train_dataset = TensorDataset(train_inputs_tensor, train_targets_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_inputs_tensor, val_targets_tensor)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    n_batches = len(train_dataloader)
    
    print(f"开始训练，总共 {n_epochs} 轮，训练样本 {len(train_inputs)} 条，验证样本 {len(val_inputs)} 条")

    for epoch in range(n_epochs):
        # 训练模式
        model.train()
        epoch_start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_dataloader:
            ### START YOUR CODE ###
            # 获取批次数据 (已经在GPU上)
            batch_inputs, batch_targets = batch
            
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ### END YOUR CODE ###

            epoch_loss += loss.item()
            batch_count += 1
            
            if batch_count % print_loss_every == 0:
                avg_loss = epoch_loss / batch_count 
                progress = batch_count / n_batches * 100
                sys.stdout.write(f'\rEpoch {epoch+1}/{n_epochs} - [{progress:.1f}%] - Loss: {avg_loss:.4f}')
                sys.stdout.flush()
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():  # 不计算梯度，节省内存
            for val_batch in val_dataloader:
                val_inputs, val_targets = val_batch  # 已经在GPU上
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_targets).item()
                
                # 计算准确率
                _, predicted = val_outputs.max(1)
                val_total += val_targets.size(0)
                val_correct += predicted.eq(val_targets).sum().item()
        
        val_loss /= len(val_dataloader)
        val_accuracy = val_correct / val_total * 100
        
        # 打印本轮结果
        train_loss = epoch_loss / len(train_dataloader)
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        
        print()
        print(f'Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Time: {epoch_time:.2f}s')
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            print(f'✓ 验证损失改善，保存最佳模型状态 (损失: {val_loss:.4f})')
        else:
            patience_counter += 1
            print(f'✗ 验证损失未改善 ({patience_counter}/{patience})')
            if patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已加载最佳模型状态")
    
    # 保存模型
    if args.model is not None:
        model_path = f'{args.model_type}_{args.model}'
    else:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'{args.model_type}_model_{now}.pt'
    
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")