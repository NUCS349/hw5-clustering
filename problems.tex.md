## Problems

## Code implementation (5 points)
Pass test cases by implementing the functions in the `code` directory.

Your grade for this section is defined by the autograder. If it says you got an 80/100,
you get 4 points here.

## Free response questions (5 points)

Answer the following free response questions in a separate document, 
saved as a .pdf and **uploaded to Canvas**.

### Polynomial regression (2 points)

First, we will run experiments on the `PolynomialRegression` code you wrote.

Generate random data (via the function you implemented `generate_regression_data`) 
and perform regression on it with the class `PolynomialRegression` that you
implemented. Generate data of varying degrees, from 1 to 10. For each generated
dataset (.1 points each):

1. Show a scatter plot of each dataset. Indicate its degree in the title of the plot.
2. Plot the polynomial of that degree on top of the scatter plot found by your implementation of 
`PolynomialRegression`.
3. Try all polynomials of degree 1:10 on the dataset. Show a plot of mean squared error as a 
function of degree.

In the plots of mean squared error vs degree, what patterns do you notice? Specifically
in relation to the actual degree polynomial used to generate the dataset. Explain why
using the highest degree polynomial possible may or may not be a good idea. Put your
answer in terms of *bias* and *variance* (1 point):

https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff

### Perceptron (2 points)

The following datasets are provided:

```
data/blobs.json
data/circles.json
data/crossing.json
data/parallel_lines.json
data/transform_me.json
```

Note that you can upload these files to the website http://ml-playground.com/. This
website will let you see how different learning algorithms work on each dataset. This
is not required for points but might be illuminating and useful for checking your work!

For each of these 5 datasets (.2 points each):

1. Show a scatter plot of the data with each point colored according to its class.
   1. See load_json_data for code example.
2. Draw the linear separator learned by your perceptron on top of the scatter plot.
   1. Not all of the datasets will be separable, so this plot may look odd.
3. Did your perceptron work on the dataset? Why or why not? 

For the dataset `data/transform_me.json` (1 point):
1. Make sure the relevant test case in the function `transform_data` in `code/perceptron.py` passes.
2. Show two scatter plots of the data before transformation and after transformation.
3. Why did the transformation you implemented work?
4. Read about the kernel method (sometimes called the kernel trick):
   1. https://en.wikipedia.org/wiki/Kernel_method
5. Explain how the transformation you implemented relates to the kernel trick.