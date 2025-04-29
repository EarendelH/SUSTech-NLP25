import sys
import numpy as np
import torch
import argparse

from model import BaseModel, WordPOSModel, BiLSTMWordPOSModel
from parse_utils import DependencyArc, DependencyTree, State, parse_conll_relation
from get_train_data import FeatureExtractor

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str, default='model.pt')
argparser.add_argument('--words_vocab', default='words_vocab.txt')
argparser.add_argument('--pos_vocab', default='pos_vocab.txt')
argparser.add_argument('--rel_vocab', default='rel_vocab.txt')


class Parser(object):
    def __init__(self, extractor: FeatureExtractor, model_file: str):
        ### START YOUR CODE ###
        # 根据模型文件确定使用的模型类型
        word_vocab_size = len(extractor.word_vocab)
        pos_vocab_size = len(extractor.pos_vocab)
        output_size = len(extractor.rel_vocab)
        
        # 判断模型类型：如果模型文件名包含'base'，则使用BaseModel，否则使用WordPOSModel
        if 'base' in model_file:
            self.model = BaseModel(word_vocab_size, output_size)
            self.model_type = 'base'
        elif 'wordpos' in model_file:
            self.model = WordPOSModel(word_vocab_size, pos_vocab_size, output_size)
            self.model_type = 'wordpos'
        else :
            self.model = BiLSTMWordPOSModel(word_vocab_size, pos_vocab_size, output_size)
            self.model_type = 'bilstm'
        
        ### END YOUR CODE ###
        
        self.model.load_state_dict(torch.load(model_file, weights_only=True))
        self.model.eval()  # 设置为评估模式
        self.extractor = extractor

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict(
            [(index, action) for (action, index) in extractor.rel_vocab.items()]
        )

    def parse_sentence(self, words, pos):
        state = State(range(1, len(words)))
        state.stack.append(0)

        while state.buffer:
            ### START YOUR CODE ###
            # 提取当前状态的表示
            if self.model_type == 'base':
                current_state = self.extractor.get_input_repr_word(words, pos, state)
            else:
                current_state = self.extractor.get_input_repr_wordpos(words, pos, state)
            
            # 转换为PyTorch张量并进行预测
            with torch.no_grad():
                input_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
                outputs = self.model(input_tensor)
                # 对输出应用softmax得到概率分布
                probs = torch.softmax(outputs, dim=1)
                # 获取最高概率的动作索引
                best_action_idx = probs.argmax(dim=1).item()
            ### END YOUR CODE ###

            ### START YOUR CODE ###
            # 将索引转换为动作
            best_action = self.output_labels[best_action_idx]
            ### END YOUR CODE ###

            ### START YOUR CODE ###
            # 根据预测的动作更新解析状态
            if best_action[0] == "shift":
                state.shift()
            elif best_action[0] == "left_arc":
                # 左弧: stack[-1] <- buffer[-1]
                state.left_arc(best_action[1])
            elif best_action[0] == "right_arc":
                # 右弧: stack[-1] -> buffer[-1]
                state.right_arc(best_action[1])
            ### END YOUR CODE ###

        ### START YOUR CODE ###
        # 根据依存关系构建依存树
        tree = DependencyTree()
        
        # 添加根节点和词汇节点
        for i in range(1, len(words)):
            # 初始化每个词为依赖弧，使用虚拟头结点0
            tree.add_deprel(DependencyArc(i, words[i], pos[i], 0, "root"))
        
        # 更新依存关系
        for head, dependent, label in state.deps:
            # 更新依赖节点的head和deprel
            for id in tree.deprels:
                if tree.deprels[id].id == dependent:
                    tree.deprels[id].head = head
                    tree.deprels[id].deprel = label
                    break
        ### END YOUR CODE ###

        return tree

if __name__ == "__main__":
    args = argparser.parse_args()
    try:
        word_vocab_file = open(args.words_vocab, "r")
        pos_vocab_file = open(args.pos_vocab, "r")
        rel_vocab_file = open(args.rel_vocab, "r")
    except FileNotFoundError:
        print(f'Could not find vocabulary files {args.words_vocab}, {args.pos_vocab}, and {args.rel_vocab}')
        sys.exit(1)
    
    extractor = FeatureExtractor(word_vocab_file, pos_vocab_file, rel_vocab_file)
    parser = Parser(extractor, args.model)

    # Test an example sentence, 3rd example from dev.conll
    words = [None, 'The', 'bill', 'intends', 'to', 'restrict', 'the', 'RTC', 'to', 'Treasury', 'borrowings', 'only', ',', 'unless', 'the', 'agency', 'receives', 'specific', 'congressional', 'authorization', '.']
    pos = [None, 'DT', 'NN', 'VBZ', 'TO', 'VB', 'DT', 'NNP', 'TO', 'NNP', 'NNS', 'RB', ',', 'IN', 'DT', 'NN', 'VBZ', 'JJ', 'JJ', 'NN', '.']

    tree = parser.parse_sentence(words, pos)
    print(tree)