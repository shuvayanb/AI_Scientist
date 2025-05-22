This repo is an fork of the original Github repo **The AI Scientist** maintained by Sakana AI [here](https://github.com/SakanaAI/AI-Scientist). This repo puts the AI Scientist to test on an entirely new problem, i.e., Sparse Identification of Nonlinear Dynamical systems (SINDy). 

<p align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/adaptive_dual_scale_denoising/adaptive_dual_scale_denoising.pdf"><img src="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/anim-ai-scientist.gif" alt="Adaptive Dual Scale Denoising" width="80%" />
</a></p>

Some of the specifications for this test cases are:

#### Model: “Anthropic API (claude-3-5-sonnet-20240620)”
#### Writeup: “LaTeX”
#### Improvement: “True”
#### Num_ideas: 1 
#### Engine: “Semantic Scholar”
#### Experiment: “SINDy”


SINDy[here](https://pysindy.readthedocs.io/en/latest/examples/2_introduction_to_sindy/example.html) is a method that discovers governing equations directly from data. It formulates the problem as a sparse regression, where a library of potential functions (e.g., polynomials, trigonometric terms) is built, and sparse regression identifies the minimal set of terms needed to represent the underlying dynamics accurately. Consequently, for the problem of choice, the parameters that needs tuning to arrive at the correct coefficients representing the underlying dynamic equation are:

PolynomialLibrary:
feature_library=ps.PolynomialLibrary(degree=poly_order)
Here, the feature_library (or dictionary) is the library of candidate functions. Common choices include polynomial terms, trigonometric functions, exponentials, and other nonlinear transformations of the state variables.
Data collection:
Gather time-series data from the system, typically in the form of measurements of the 
state variables at discrete time intervals. These serve as the training data with which 
sparse regression is performed. 
Computing derivatives:
Numerically estimate time derivatives from the data, e.g., via finite differences or smoothing methods.

The above choice (besides the choice of ODE class) becomes the parameters that AI Scientist can tune and generate novel ideas on. On the same vein, AI Scientist requires a fixed template to be generated for a new use case, as shown in Fig. below. 

<img width="850" alt="sakana" src="https://github.com/user-attachments/assets/7e1c4505-18c5-4535-b3cc-050e25a0c88e" />

The report: the full 10 page paper written up by AI Scientist at the end of this is shown below

<img width="891" alt="sakana2" src="https://github.com/user-attachments/assets/6fc2c4dc-bcf7-4dcb-aa3a-709d85a34dbb" />

