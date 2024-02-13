# Homework 2
- The entire assignment is done in the file `all_questions.py`. 
- Do not change the name of the file. 
- You can run python on this file to check for compilation errors. Do so BEFORE uploading to Gradescope. 
```code
    python all_questions.py
```
- Each question is answered using a dictionary that is set up in the various functions in the file `all_questions.py`
- There are 7 questions. 
- Do not simply fill the answers with numbers. I require the the work to produce these answers
  be included in the code. You are allowed to count classes and labels from the table or via other means. 
  You are not required to use Pandas. But the answers must be derived via formulas when possible. 
  For example, entropy and Gini should be calculated. I expect floats to be correct to within 0.01 percent. 
  That means that 0.5232 and 0.5233 are considered to be different. 
  - For example, if you have a binary classification problem with labels 'yes'/'no' , and an attribute named "X" 
  (married/single), you could defined variables: `nb_n` and `nb_y`, and then compute the entropy as:
  ```
  nb_y = 4;   nb_n = 6
  ntot = nb_y + nb_n
  H = -(nb_y/ntot * u.log2(nb_y/ntot) + nb_n/ntot * u.log2(nb_n/ntot))
  ```
  This is just an example for illustration. 
- Some answers require explanations, which you should add to your answers as a string. 
- for the next 10 days, you can submit to gradescope, which will do structural checks
    - make sure there are no missing keys
    - check that the answer types (float/string) are correct
- The answers themselves, which are graded, are checked after the due date has passed. 

### Report any errors or questions on the discussion area on Canvas for Homework 2. 
