

critic_judge_prompt = """\
I will provide you with a question and a robot’s answer, divided into several sections. Your task is to evaluate each section of the robot’s answer for any errors.

**Evaluation Criteria:**
- Evaluate each section independently. Assess each section based solely on the accuracy and logic within that section.
- Do not consider subjective elements such as redundancy or stylistic preferences as errors.
- Do not consider corrections or reflections made in later sections. Even if a later section acknowledges and fixes an earlier mistake, the original section must still be marked as erroneous.
- If a subsequent section contains an error caused by an earlier section's mistake, do not count it as a new error.

**Output Format:**
- If you think all sections of the robot's answer are correct, output in the following format:  
    Conclusion: no error
- If you think any section contains an error, output in the following format:  
    Conclusion: yes
    Error Section Number: [section number]
    Explanation: [explanation for the error in this section]
    Error Section Number: [section number]
    Explanation: [explanation for the error in this section]
    … (repeat for each erroneous section)  


**example:**
- Question: Two numbers, $x$ and $y$ are selected at random from the interval $(0,3)$.  What is the probability that a triangle with sides of length 1, $x$, and $y$ exists?
- Robot's Answer: 
section1:
So I've got this probability problem here. It says that two numbers, x and y, are selected at random from the interval (0,3), and I need to find the probability that a triangle with sides of length 1, x, and y exists. Okay, let's break this down.

section2:
First, I need to recall the triangle inequality theorem. It states that for any triangle with sides a, b, and c, the following must be true:
1. a + b > c
2. a + c > b
3. b + c > a

section3:
So, in this case, the sides are 1, x, and y. That means I need to satisfy all three inequalities:
1. 1 + x > y
2. 1 + y > x
3. x + y > 1
Alright, now, to find the probability, I need to find the ratio of the area where these inequalities hold to the total area from which x and y are selected.

section4:
Now, I need to find the area where all three inequalities are satisfied.
Let me consider each inequality one by one.
First inequality: 1 + x > y
This can be rewritten as y < 1 + x.
Second inequality: 1 + y > x
This can be rewritten as y > x - 1.
Third inequality: x + y > 1
This can be rewritten as y > 1 - x.
Now, since x and y are both positive (from (0,3)), 1 - x could be negative if x > 1. But since y is positive, y > 1 - x will always hold true when x < 1, and when x >=1, y > 1 - x is always true because y > 0 and 1 - x <=0. So, this inequality is always satisfied in the given range. Therefore, I don't need to worry about this inequality separately.

section5:
So, effectively, I only need to consider the first two inequalities:
1. y < 1 + x
2. y > x - 1
And, of course, x and y are between 0 and 3.

- judge result: 
Conclusion: yes
Error Section Number: 4
Explanation: The analysis of the inequality \( x + y > 1 \) is incorrect. The argument that "this inequality is always satisfied in the given range" is flawed. Specifically, for small values of \( x \) and \( y \), such as when both are close to zero, the inequality \( x + y > 1 \) does not always hold. The constraint \( x + y > 1 \) must still be explicitly considered in the probability calculation.

**Input:**
- Question: {{question}}
- Robot's Answer: {{model_output}}
- judge result: \
"""