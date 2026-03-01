# Project Overview
This is a black-box(BBO) capstone project to maximise the global optimum for eight functions whose objective functions are unknown. This is an optimisation challenge. The process of discovering, unearthing, searching etc. often begins with exploring unknowns until we either prove or disprove the presence of the our object of interest. This challenge has likewise objectives; iteratively enquire until we “strike gold”. This technique is used for minerals prospecting, oil field discovery, hyperparameter optimisation and tuning, detecting radioactivity levels etc. The high level idea is: without an explicit objective function or a direct formula which underlies our model can we optimise this function? 

In this project, I will be using surrogate models to approximate the objective function and guide the optimisation process. The goal is to maximise all eight functions based on the outputs we observe, even though the underlying function remains opaque throughout the optimisation cycles. 

Using a mix of exploration and exploitation, I’ll carefully blend these two strategies to discover new search spaces and dig deeper into promising, high yield areas. Bayesian optimisation will be the primary tool of choice. Acquisition functions will be utilised to handle uncertainty and make informed decisions on where to evaluate the next function.

# Inputs and Outputs
The model receives continuous, random variables(6 decimal places) as input for all eight functions of varying dimensions and returns a scalar, continuous variable. Input parameters are within the range of 0 and 1. 

Here’s an example of the expected input. 

|**Function**|Dimensions|Typical input values|
|:-|:-|:-|
|1|2| 0.516192-0.628562|
|2|2| 0.819027-0.999999|
|3|3| 0.999999-0.999999-0.999999|
|4|3| 0.411215-0.391663-0.334151-0.430246|
|5|3| 0.507329-0.771394-0.529443-0.567015|
|6|5| 0.000001-0.000001-0.000001-0.999999-0.000001|
|7|6| 0.045149-0.302222-0.352880-0.146446-0.345691-0.763203|
|8|8| 0.000001-0.011975-0.157238-0.000001-0.999999-0.464723-0.000001-0.780942|

These values are submitted to the unknown, black box function which returns scalar values as output for each of the functions. 

# Challenge Objectives
Maximisation is the goal for all eight functions. 

# Technical Approach
The first three cycles for generating the next input variables for each of the eight functions utilised three distinct variations of the same acquisition strategy, Upper Continuous Bounds(UCB). Future iterations are expected to utilise Expected Improvements(EI) as the acquisition strategy to exploit potential high yield areas. This is a Bayesian, sequential approach that uses prior results to improve input suggestions and minimise the effect of false positives (low success data points). There’s no early stopping mechanism. The challenge ends after twelve sequences of the Bayesian optimisation cycle.

The first cycle’s input were randomly generated values. These random variables were constrained to be within the exclusive bounds of 0 to 1. Gaussian Processes (GP) was used as a surrogate model. This marked an early change of strategy from randomly generated data as this strategy did not intelligently shape the next set of input data. In addition, I pivoted early from guessing subsequent input data as surrogate models can help manage complexity and provide a more structured, repeatable approach to evaluating the true objective function. I used GPs to build the surrogate model as they provide excellent statistical properties and uncertainty estimates. This is in contrast to other potential surrogate function candidates such as regression trees etc. Uncertainty estimates help to refine strategies for exploration or exploitation. 
Bayesian techniques, which GPs are, helps me to handle uncertainty in the model predictions as I update beliefs with new data. SVM models may be implemented when we can distinctly decide on the classes of interest. In this early stages, there isn’t enough data to delineate and categorise classes with labels with helpful labels i.e. high and low performance regions. 

The early stages of the process is dedicated to exploring unknown regions. The strategy will be adapted new cycles expose more information to help drive exploitation of potentially, profitable regions. 