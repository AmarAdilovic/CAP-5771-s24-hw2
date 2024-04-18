# Answer found in Q5 in Question Bank 1 (Tan et al, 2nd ed)

# import student_code_with_answers.utils as u
import utils as u
import pprint

# Example of how to specify a binary with the structure:
# See the file INSTRUCTIONS.md
# ----------------------------------------------------------------------
"""
Consider the training set given below for predicting lung cancer in patients based on their symptoms (chronic cough and weight loss)
and other lifestyle and environmental attributes (tobacco smoking and exposure to radon).

Encode the two-level decision tree into 'utils.BinaryTree' obtained using entropy as the impurity measure.

Fill the answer dictionary provided in the python function provided.
Show your steps in code for every step of the tree construction process.

Compute the training error of the decision tree
"""
def question1():
    """
    Note 1: Each attribute can appear as a node in the tree AT MOST once.
    Note 2: For level two, fill the keys for all cases left and right.
    If an attribute is not considered for level 2, set the values to -1.
    For example, if "flu" were the choice for level 1 (it is not), then set level2_left['flu'] = level2_right['flu'] = -1.,
    and the same for keys 'flu_info_gain'.
    """
    answer = False
    answer = {}
    level1 = {}
    level2_left = {}
    level2_right = {}

    # to calculate entropy from the textbook:
    # -sum(pi(t) log2 pi(t))
    #   where pi(t) is the relative frequency of training instances that belong to class i at node t,
    #   c is the total number of classes,
    #   and 0 log2 0 = 0
    #  for a binary classification problem: p0(t) + p1(t) = 1

    """
    Tobacco Radon       Chronic   Weight    Lung
    Smoking Exposure    Cough     Loss     Cancer
    Yes     Yes         Yes       No        Yes
    Yes     No          Yes       No        Yes
    Yes     No          Yes       Yes       Yes
    Yes     No          Yes       Yes       Yes
    No      Yes         No        Yes       Yes
    Yes     No          No        No        No
    No      No          Yes       No        No
    No      No          Yes       Yes       No
    No      No          Yes       No        No
    No      No          No        Yes       No 
    """
    def compute_entropy(num_y, num_n):
        if (num_n == 0 or num_y == 0): return 0
        total = num_y + num_n
        calculated_entropy = (-(num_y / total) * u.log2(num_y / total)) - ((num_n / total) * u.log2(num_n / total))
        return calculated_entropy

    def sum_entropy(num_y, entropy_y, num_n, entropy_n):
        total = num_y + num_n
        return  ((num_y / total) * entropy_y) + ((num_n / total) * entropy_n)

    # target_entropy is the entropy of the "Lung Cancer" column
    def compute_info_gain(entropy, target_entropy = compute_entropy(num_y=5, num_n=5)):
        return target_entropy - entropy

    smoking_entropy_y = compute_entropy(num_y=1, num_n=4)
    smoking_entropy_n = compute_entropy(num_y=4, num_n=1)
    smoking_entropy = sum_entropy(num_y=7, entropy_y=smoking_entropy_y, num_n=3, entropy_n=smoking_entropy_n)
    level1["smoking"] = smoking_entropy
    smoking_info_gain = compute_info_gain(smoking_entropy)
    level1["smoking_info_gain"] = smoking_info_gain

    cough_entropy_y = compute_entropy(num_y=4, num_n=3)
    cough_entropy_n = compute_entropy(num_y=1, num_n=2)
    cough_entropy = sum_entropy(num_y=7, entropy_y=cough_entropy_y, num_n=3, entropy_n=cough_entropy_n)
    level1["cough"] = cough_entropy
    cough_info_gain = compute_info_gain(cough_entropy)
    level1["cough_info_gain"] = cough_info_gain

    radon_entropy_y = compute_entropy(num_y=2, num_n=0)
    radon_entropy_n = compute_entropy(num_y=3, num_n=5)
    radon_entropy = sum_entropy(num_y=2, entropy_y=radon_entropy_y, num_n=8, entropy_n=radon_entropy_n)
    level1["radon"] = radon_entropy
    radon_info_gain = compute_info_gain(radon_entropy)
    level1["radon_info_gain"] = radon_info_gain

    weight_loss_entropy_y = compute_entropy(num_y=2, num_n=3)
    weight_loss_entropy_n = compute_entropy(num_y=3, num_n=2)
    weight_loss_entropy = sum_entropy(num_y=5, entropy_y=weight_loss_entropy_y, num_n=5, entropy_n=weight_loss_entropy_n)
    level1["weight_loss"] = weight_loss_entropy
    weight_loss_info_gain = compute_info_gain(weight_loss_entropy)
    level1["weight_loss_info_gain"] = weight_loss_info_gain

    # Smoking has the highest information gain, starting with yes

    # If an attribute is not considered for level 2, set the values to -1.
    level2_left["smoking"] = -1.0
    level2_left["smoking_info_gain"] = -1.0
    level2_right["smoking"] = -1.0
    level2_right["smoking_info_gain"] = -1.0

    level2_left_radon_entropy_y = compute_entropy(num_y=1, num_n=0)
    level2_left_radon_entropy_n = compute_entropy(num_y=3, num_n=1)
    level2_left_radon_entropy = sum_entropy(num_y=1, entropy_y=level2_left_radon_entropy_y, num_n=4, entropy_n=level2_left_radon_entropy_n)
    level2_left["radon"] = level2_left_radon_entropy
    level2_left_radon_info_gain = compute_info_gain(level2_left_radon_entropy, target_entropy=smoking_entropy)
    level2_left["radon_info_gain"] = level2_left_radon_info_gain

    level2_left_cough_entropy_y = compute_entropy(num_y=4, num_n=0)
    level2_left_cough_entropy_n = compute_entropy(num_y=0, num_n=1)
    level2_left_cough_entropy = sum_entropy(num_y=4, entropy_y=level2_left_cough_entropy_y, num_n=1, entropy_n=level2_left_cough_entropy_n)
    level2_left["cough"] = level2_left_cough_entropy
    level2_left_cough_info_gain = compute_info_gain(level2_left_cough_entropy, target_entropy=smoking_entropy)
    level2_left["cough_info_gain"] = level2_left_cough_info_gain

    level2_left_weight_loss_entropy_y = compute_entropy(num_y=2, num_n=0)
    level2_left_weight_loss_entropy_n = compute_entropy(num_y=2, num_n=1)
    level2_left_weight_loss_entropy = sum_entropy(num_y=2, entropy_y=level2_left_weight_loss_entropy_y, num_n=3, entropy_n=level2_left_weight_loss_entropy_n)
    level2_left["weight_loss"] = level2_left_weight_loss_entropy
    level2_left_weight_loss_info_gain = compute_info_gain(level2_left_weight_loss_entropy, target_entropy=smoking_entropy)
    level2_left["weight_loss_info_gain"] = level2_left_weight_loss_info_gain

    # Now evaluating no

    level2_right_radon_entropy_y = compute_entropy(num_y=1, num_n=0)
    level2_right_radon_entropy_n = compute_entropy(num_y=0, num_n=4)
    level2_right_radon_entropy = sum_entropy(num_y=1, entropy_y=level2_right_radon_entropy_y, num_n=4, entropy_n=level2_right_radon_entropy_n)
    level2_right["radon"] = level2_right_radon_entropy
    level2_right_radon_info_gain = compute_info_gain(level2_right_radon_entropy, target_entropy=smoking_entropy)
    level2_right["radon_info_gain"] = level2_right_radon_info_gain

    level2_right_cough_entropy_y = compute_entropy(num_y=3, num_n=0)
    level2_right_cough_entropy_n = compute_entropy(num_y=1, num_n=1)
    level2_right_cough_entropy = sum_entropy(num_y=3, entropy_y=level2_right_cough_entropy_y, num_n=2, entropy_n=level2_right_cough_entropy_n)
    level2_right["cough"] = level2_right_cough_entropy
    level2_right_cough_info_gain = compute_info_gain(level2_right_cough_entropy, target_entropy=smoking_entropy)
    level2_right["cough_info_gain"] = level2_right_cough_info_gain

    level2_right_weight_loss_entropy_y = compute_entropy(num_y=0, num_n=2)
    level2_right_weight_loss_entropy_n = compute_entropy(num_y=1, num_n=2)
    level2_right_weight_loss_entropy = sum_entropy(num_y=2, entropy_y=level2_right_weight_loss_entropy_y, num_n=3, entropy_n=level2_right_weight_loss_entropy_n)
    level2_right["weight_loss"] = level2_right_weight_loss_entropy
    level2_right_weight_loss_info_gain = compute_info_gain(level2_right_weight_loss_entropy, target_entropy=smoking_entropy)
    level2_right["weight_loss_info_gain"] = level2_right_weight_loss_info_gain

    answer["level1"] = level1
    answer["level2_left"] = level2_left
    answer["level2_right"] = level2_right

    # Fill up `construct_tree``
    # tree, training_error = construct_tree()
    tree = u.BinaryTree("x < 0.5")  # MUST STILL CREATE THE TREE *****
    answer["tree"] = tree  # use the Tree structure

    # answer["training_error"] = training_error
    answer["training_error"] = 0.0  
    # print("Question 1: ")
    # pprint.pprint(answer)

    return answer


# ----------------------------------------------------------------------
"""
Consider a training set sampled uniformly from the two-dimensional space shown in Figure 1.
Assume that the training set size is large enough so that the probabilities can be calculated accurately based on the areas of the selected regions.

The space is divided into three classes A, B, and C.
In this exercise, you will build a decision tree from the training set.
"""

def question2():
    answer = {}

    # to calculate entropy from the textbook:
    # -sum(pi(t) log2 pi(t))
    #   where pi(t) is the relative frequency of training instances that belong to class i at node t,
    #   c is the total number of classes,
    #   and 0 log2 0 = 0

    # Class area proportion (two rectangles, height * width + h * w)
    class_A_proportion = (0.4 * 0.8) + (0.3 * 0.3)
    class_B_proportion = (0.6 * 0.7) + (0.2 * 0.2)
    class_C_proportion = (0.3 * 0.3) + (0.2 * 0.2)

    total = class_A_proportion + class_B_proportion + class_C_proportion

    def compute_entropy(proportions, total=total):
        return -sum((p / total) * u.log2((p / total)) for p in proportions if p > 0)


    overall_entropy = compute_entropy([class_A_proportion, class_B_proportion, class_C_proportion])
    # Answers are floats
    # (a) Compute the entropy for the overall data
    answer["(a) entropy_entire_data"] = overall_entropy

    # Compute the entropy for each split
    # Split at x ≤ 0.2
    class_A_proportion = 0
    class_B_proportion = (0.6 * 0.2) + (0.2 * 0.2)
    class_C_proportion = (0.2 * 0.2)
    split = class_A_proportion + class_B_proportion + class_C_proportion

    split_x_02_entropy = compute_entropy([class_A_proportion, class_B_proportion, class_C_proportion], total=(split))

    # Split at x ≤ 0.7
    class_A_proportion = (0.4 * 0.5)
    class_B_proportion = (0.6 * 0.7) + (0.2 * 0.2)
    class_C_proportion = (0.2 * 0.2)
    split = class_A_proportion + class_B_proportion + class_C_proportion

    split_x_07_entropy = compute_entropy([class_A_proportion, class_B_proportion, class_C_proportion], total=(split))

    # Split at y ≤ 0.6
    class_A_proportion = (0.3 * 0.3)
    class_B_proportion = (0.6 * 0.7)
    class_C_proportion = (0.3 * 0.3)
    split = class_A_proportion + class_B_proportion + class_C_proportion

    split_y_06_entropy = compute_entropy([class_A_proportion, class_B_proportion, class_C_proportion], total=(split))


    # Infogain
    # (b) Compare the entropy when the data is split at x ≤ 0.2, x ≤ 0.7, and y ≤ 0.6.
    answer["(b) x <= 0.2"] =  overall_entropy - (split_x_02_entropy)
    answer["(b) x <= 0.7"] = overall_entropy - (split_x_07_entropy)
    answer["(b) y <= 0.6"] = overall_entropy - (split_y_06_entropy)

    # choose one of 'x=0.2', 'x=0.7', or 'x=0.6'
    # (c) Based on your answer in part (b), which attribute split condition should be used as the root of the decision tree.
    answer["(c) attribute"] = "x=0.2"  

    # Use the Binary Tree structure to construct the tree
    # Answer is an instance of BinaryTree
    # (d) Draw the full decision tree for the data set.
    # Constraints: Use only the decision boundaries x = 0.7, x = 0.2, y = 0.6, y = 0.3, y = 0.8 to contruct your tree.
    # Each level of the tree will maximize the information gain.
    # Use the utils.BinaryTree class to save the tree in the ‘answer‘ dictionary
    tree = u.BinaryTree("x <= 0.2")
    
    answer["(d) full decision tree"] = tree
    # print("Question 2: ")
    # pprint.pprint(answer)

    return answer


# ----------------------------------------------------------------------


def question3():
    answer = {}

    # float
    answer["(a) Gini, overall"] = 0.

    # float
    answer["(b) Gini, ID"] = 0.0
    answer["(c) Gini, Gender"] = 0.
    answer["(d) Gini, Car type"] = 0.
    answer["(e) Gini, Shirt type"] = 0.

    answer["(f) attr for splitting"] = ""

    # Explanatory text string
    answer["(f) explain choice"] = ""

    return answer


# ----------------------------------------------------------------------
# Answers in th form [str1, str2, str3]
# If both 'binary' and 'discrete' apply, choose 'binary'.
# str1 in ['binary', 'discrete', 'continuous']
# str2 in ['qualitative', 'quantitative']
# str3 in ['interval', 'nominal', 'ratio', 'ordinal']


def question4():
    answer = {}

    # [string, string, string]
    # Each string is one of ['binary', 'discrete', 'continuous', 'qualitative', 'nominal', 'ordinal',
    #  'quantitative', 'interval', 'ratio'
    # If you have a choice between 'binary' and 'discrete', choose 'binary'

    # (a) Time in terms of AM or PM.
    answer["a"] = ['binary', 'qualitative', 'ordinal']

    # Explain if there is more than one interpretation. Repeat for the other questions. At least five words that form a sentence.
    answer["a: explain"] = "It is binary as there are two categories, AM or PM. Could be discrete if ever decreasing measurements of time are ignored."

    # (b) Brightness as measured by a light meter.
    answer["b"] = ['continuous', 'quantitative', 'ratio']
    answer["b: explain"] = ""

    # (c) Brightness as measured by peoples judgments.
    answer["c"] = ['discrete', 'qualitative', 'ordinal']
    answer["c: explain"] = "People will categorize brightness into very bright, bright, and not bright generally, with no equal interval between the values."

    # (d) Angles as measured in degrees between 0 and 360.
    answer["d"] = ['continuous', 'quantitative', 'ratio']
    answer["d: explain"] = "Could be interval, but the defined zero point points to ratio."

    # (e) Bronze, Silver, and Gold medals as awarded at the Olympics.
    answer["e"] = ['discrete', 'qualitative', 'ordinal']
    answer["e: explain"] = ""

    # (f) Height above sea level.
    answer["f"] = ['continuous', 'quantitative', 'interval']
    answer["f: explain"] = "Could be ratio but as it is above sea level it implies there is no zero point"

    # (g) Number of patients in a hospital.
    answer["g"] = ['discrete', 'quantitative', 'ratio']
    answer["g: explain"] = "Could be interval however as there could be a defined zero, ratio."

    # (h) ISBN numbers for books. (Look up the format on the Web.)
    answer["h"] = ['discrete', 'quantitative', 'nominal']
    answer["h: explain"] = ""

    # (i) Ability to pass light in terms of the following values: opaque, translucent, transparent.
    answer["i"] = ['discrete', 'qualitative', 'ordinal']
    answer["i: explain"] = ""

    # (j) Military rank.
    answer["j"] = ['discrete', 'qualitative', 'ordinal']
    answer["j: explain"] = ""

    # (k) Distance from the center of campus.
    answer["k"] = ['continuous', 'quantitative', 'ratio']
    answer["k: explain"] = ""

    # (l) Density of a substance in grams per cubic centimeter.
    answer["l"] = ['continuous', 'quantitative', 'ratio']
    answer["l: explain"] = ""

    # (m) Coat check number. (When you attend an event, you can often give
    # your coat to someone who, in turn, gives you a number that you can
    # use to claim your coat when you leave.)
    answer["m"] = ['discrete', 'qualitative', 'nominal']
    answer["m: explain"] = ""

    return answer


# ----------------------------------------------------------------------


def question5():
    explain = {}

    # Read appropriate section of book chapter 3

    # string: one of 'Model 1' or 'Model 2'
    explain["a"] = ""
    explain["a explain"] = ""

    # string: one of 'Model 1' or 'Model 2'
    explain["b"] = ""
    explain["b explain"] = ""

    explain["c similarity"] = ""
    explain["c similarity explain"] = ""

    explain["c difference"] = ""
    explain["c difference explain"] = ""

    return explain


# ----------------------------------------------------------------------
def question6():
    answer = {}
    # x <= ? is the left branch
    # y <= ? is the left branch

    # value of the form "z <= float" where "z" is "x" or "y"
    #  and "float" is a floating point number (notice: <=)
    # The value could also be "A" or "B" if it is a leaf
    answer["a, level 1"] = ""
    answer["a, level 2, right"] =""
    answer["a, level 2, left"] = ""
    answer["a, level 3, left"] = ""
    answer["a, level 3, right"] = ""

    # run each datum through the tree. Count the number of errors and divide by number of samples. .
    # Since we have areas: calculate the area that is misclassified (total area is unity)
    # float between 0 and 1
    answer["b, expected error"] = 0.

    # Use u.BinaryTree to define the tree. Create your tree.
    # Replace "root node" by the proper node of the form "z <= float"
    tree = u.BinaryTree("root note")

    answer["c, tree"] = tree

    return answer


# ----------------------------------------------------------------------
def question7():
    answer = {}

    # float
    answer["a, info gain, ID"] = 0.
    answer["b, info gain, Handedness"] = 0.

    # string: "ID" or "Handedness"
    answer["c, which attrib"] = ""

    # answer is a float
    answer["d, gain ratio, ID"] = 0.
    answer["e, gain ratio, Handedness"] = 0.

    # string: one of 'ID' or 'Handedness' based on gain ratio
    # choose the attribute with the largest gain ratio
    answer["f, which attrib"] = ""

    return answer


# ----------------------------------------------------------------------

if __name__ == "__main__":
    answers = {}
    answers["q1"] = question1()
    answers["q2"] = question2()
    answers["q3"] = question3()
    answers["q4"] = question4()
    answers["q5"] = question5()
    answers["q6"] = question6()
    answers["q7"] = question7()

    u.save_dict("answers.pkl", answers)

