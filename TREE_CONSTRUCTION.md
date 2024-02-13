# How to use utils.BinaryTree. 

# Example of how to specify a binary with the structure:
#
#              C
#             /  \
#            /    \
#           /      \
#          /        \
#         A          B
#        / \        / \
#       /   \      /   \
#     "y"   "n"  "n"   "y"
#
# where C = "x <= 13.0"
# where A = "y <= 10.0"
# where B = "x <= 7.3
# Assume two classes: "y" and "no"
# Leaves are always strings

import utils as u
tree = u.BinaryTree("x <= 13.0")
A = tree.insert_left("y <= 10")
B = tree.insert_right("x <= 7.3")
# Four leaves
A.insert_left("y")
A.insert_right("n")
B.insert_left("n")
B.insert_right("y")

answer["tree"] = tree

# Example of how to specify a binary with the structure:
#
#              A
#             /  \
#            /    \
#           /      \
#          /        \
#        "0"         B
#                   / \
#                  /   \
#                "1"   "0"
#
# where A = "single"
# single: "yes"  (left branch)
# single: "no" (right branch)
# Leaves are always strings
#
# where B = "age <= 30"
# 
import utils as u
tree = u.BinaryTree("single")
tree.insert_left("0")
b = tree.insert_right("age <= 30")
b.insert_left("1")
b.insert_right("0")

# FOLLOW INSTRUCTIONS!

