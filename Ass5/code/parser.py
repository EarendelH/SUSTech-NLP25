import sys
import numpy as np
import torch
import argparse

from model import BaseModel, WordPOSModel
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
        # TODO: Initialize the model
        self.model = None
        ### END YOUR CODE ###
        self.model.load_state_dict(torch.load(model_file, weights_only=True))
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
            # TODO: Extract the current state representation and make a prediction
            # Call self.extractor.get_input_repr_*()
            current_state = None
            with torch.no_grad():
                input_tensor = None # Convert current_state (np.array) to torch.tensor
                prediction = None # Call self.model()
            ### END YOUR CODE ###

            ### START YOUR CODE ###
            # TODO: Select the best action for the current state, using greedy decoding, i.e., select the action with the highest score in prediction
            # Hint: best_action should be a value of self.output_labels, i.e., a (action, relation) tuple
            best_action = None 
            ### END YOUR CODE ###

            ### START YOUR CODE ###
            # TODO: Apply the best action to the state
            # Hint: Call shift() or left_arc() or right_arc() accordingly
            if best_action[0] == "shift":
                pass
            elif best_action[0] == "left_arc":
                pass
            elif best_action[0] == "right_arc":
                pass
            ### END YOUR CODE ###

        ### START YOUR CODE ###
        # TODO: Go through each relation in state.deps and add it to the tree by calling tree.add_deprel()
        tree = DependencyTree()
        pass
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