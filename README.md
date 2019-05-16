# HW6: Clustering - EECS 349 @ NU
**IMPORTANT: PUT YOUR NETID IN THE FILE** `netid` in the root directory of the assignment. 
This is used to put the autograder output into Canvas. Please don't put someone else's netid 
here, we will check.

In this assignment, you will:
- Implement K-Means
- Implement GMM
- Apply the two approaches to a dataset to discover patterns in an unsupervised way.
  
## Clone this repository

To clone this repository run the following command:

``git clone https://github.com/nucs349/hw6-clustering-[your_username]``

`[your_username]` is replaced in the above link by your Github username. Alternatively, just look at the link in your address bar if you're viewing this README in your submission repository in a browser. Once cloned, `cd` into the cloned repository. Every assignment has some files that you edit to complete it. 

## Files you edit

See problems.md for what files you will edit and what goes into the write-up.

Do not edit anything in the `tests` directory. Files can be added to `tests` but files that exist already cannot be edited. Modifications to tests will be checked for.

## Environment setup

Make a conda environment for this assignment and then run:

``pip install -r requirements.txt``

## Running the test cases

The test cases can be run with:

``python -W ignore -m pytest -s .``

at the root directory of the assignment repository.  NOTE: this command is slightly changed
to suppress warnings that come from scikit-learn when we use adjusted_mutual_info.

## Questions? Problems? Issues?

Simply open an issue on the starter code repository for this assignment [here](https://github.com/NUCS349/hw6-clustering/issues). Someone from the teaching staff will get back to you through there!