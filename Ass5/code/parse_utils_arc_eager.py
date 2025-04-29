import sys
import copy
from collections import defaultdict
from typing import List, Tuple


class DependencyArc(object):
    """
    Represent a single dependency arc:
    """
    def __init__(self, word_id, word, pos, head, deprel):
        self.id = word_id
        self.word = word
        self.pos = pos
        self.head = head
        self.deprel = deprel
    
    def __str__(self) -> str:
        return "{d.id}\t{d.word}\t_\t_\t{d.pos}\t_\t{d.head}\t{d.deprel}\t_\t_".format(d=self)


def parse_conll_relation(s):
    fields = s.split("\t")
    word_id_str, word, lemma, upos, pos, feats, head_str, deprel, deps, misc = fields
    word_id = int(word_id_str)
    head = int(head_str)
    return DependencyArc(word_id, word, pos, head, deprel)


class DependencyTree(object):
    def __init__(self):
        self.deprels = {}
        self.root = None
        self.parent_to_children = defaultdict(list)

    def add_deprel(self, deprel):
        self.deprels[deprel.id] = deprel
        self.parent_to_children[deprel.head].append(deprel.id)
        if deprel.head == 0:
            self.root = deprel.id

    def __str__(self):
        deprels = [v for (k, v) in sorted(self.deprels.items())]
        return "\n".join(str(deprel) for deprel in deprels)
    
    def print_tree(self, parent=None):
        if not parent:
            return self.print_tree(parent=self.root)

        if self.deprels[parent].head == parent:
            return self.deprels[parent].word

        children = [self.print_tree(child) for child in self.parent_to_children[parent]]
        child_str = " ".join(children)
        return "({} {})".format(self.deprels[parent].word, child_str)

    def words(self):
        return [None] + [x.word for (i, x) in self.deprels.items()]

    def pos(self):
        return [None] + [x.pos for (i, x) in self.deprels.items()]
    
    def from_string(s):
        dtree = DependencyTree()
        for line in s.split("\n"):
            if line:
                dtree.add_deprel(parse_conll_relation(line))
        return dtree


def conll_reader(input_file):
    current_deps = DependencyTree()
    while True:
        line = input_file.readline().strip()
        if not line and current_deps:
            yield current_deps
            current_deps = DependencyTree()
            line = input_file.readline().strip()
            if not line:
                break
        current_deps.add_deprel(parse_conll_relation(line))


class State(object):
    def __init__(self, sentence=[]):
        self.stack = []
        self.buffer = []
        if sentence:
            self.buffer = list(reversed(sentence))
        self.deps = set()
        self.heads = {}  

    def shift(self):
        self.stack.append(self.buffer.pop())

    def left_arc(self, label):
        child = self.stack.pop()
        head = self.buffer[-1]
        self.deps.add((head, child, label))
        self.heads[child] = head

    def right_arc(self, label):
        head = self.stack[-1]
        child = self.buffer.pop()
        self.deps.add((head, child, label))
        self.heads[child] = head
        self.stack.append(child)  # 注意：这里与Arc-Standard不同，我们将child移到栈上

    def reduce(self):
        self.stack.pop()

    def __repr__(self):
        return "{},{},{}".format(self.stack, self.buffer, self.deps)



class RootDummy(object):
    def __init__(self):
        self.head = None
        self.id = 0
        self.deprel = None
    def __repr__(self):
        return "<ROOT>"

def get_training_instances(dep_tree: DependencyTree) -> List[Tuple[State, Tuple[str, str]]]:
    deprels = dep_tree.deprels
    sorted_nodes = [k for k, v in sorted(deprels.items())]
    state = State(sorted_nodes)
    state.stack.append(0)  # 添加ROOT节点到栈中

    # 记录每个词的头节点和依赖关系
    head_map = {}
    deprel_map = {}
    for ident, node in deprels.items():
        head_map[ident] = node.head
        deprel_map[ident] = node.deprel

    seq = []
    while state.buffer or len(state.stack) > 1:
        if len(state.stack) <= 1 and state.buffer:
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()
            continue

        if not state.buffer:
            seq.append((copy.deepcopy(state), ("reduce", None)))
            state.reduce()
            continue

        stack_id = state.stack[-1]
        buffer_id = state.buffer[-1]
        
        if head_map.get(stack_id) == buffer_id and stack_id not in state.heads:
            seq.append((copy.deepcopy(state), ("left_arc", deprel_map[stack_id])))
            state.left_arc(deprel_map[stack_id])
        
        elif head_map.get(buffer_id) == stack_id:
            seq.append((copy.deepcopy(state), ("right_arc", deprel_map[buffer_id])))
            state.right_arc(deprel_map[buffer_id])
        
        elif stack_id in state.heads:
            seq.append((copy.deepcopy(state), ("reduce", None)))
            state.reduce()
        
        else:
            seq.append((copy.deepcopy(state), ("shift", None)))
            state.shift()

    return seq