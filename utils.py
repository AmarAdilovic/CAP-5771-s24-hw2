import math
import pickle


def log2(x):
    return math.log(x, 2)

def save_dict(filenm, dct):
    with open(filenm, "wb") as file:
        pickle.dump(dct, file)


# Loading from a pickle file
def load_dict(filenm, dct):
    with open(filenm, "rb") as file:
        loaded_data = pickle.load(file)
        return loaded_data
    raise "load_dict:: Error loading data"


class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert_left(self, value):
        if self.left is None:
            self.left = BinaryTree(value)
            return self.left
        else:
            new_node = BinaryTree(value)
            new_node.left = self.left
            self.left = new_node
            return new_node

    def insert_right(self, value):
        if self.right is None:
            self.right = BinaryTree(value)
            return self.right
        else:
            new_node = BinaryTree(value)
            new_node.right = self.right
            self.right = new_node
            return new_node

    def __repr__(self):
        return f"BinaryTree({self.value})"

    def __str__(self, level=0):
        ret = "\t" * level + repr(self) + "\n"
        if self.left is not None:
            ret += self.left.__str__(level + 1)
        if self.right is not None:
            ret += self.right.__str__(level + 1)
        return ret

    def print_tree(self):
        print(self.__str__())


# Example on how to create a binary tree
# A has two children: B and C
# B has two children: D and E
# C has two children: F and G
# Construct the binary tree:
def construct_binary_tree():
    root = BinaryTree("A")
    root.insert_left("B")
    root.insert_right("C")
    root.left.insert_left("D")
    root.left.insert_right("E")
    root.right.insert_left("F")
    root.right.insert_right("G")
    return root
