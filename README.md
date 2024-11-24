

---

# Gradient Descent for Multiple Linear Regression 🚀

## Overview 📝

This project provides a Python implementation of **Gradient Descent** for optimizing **Multiple Linear Regression**. It demonstrates how this optimization algorithm is used to minimize the **Mean Squared Error (MSE)** cost function to find the best-fitting regression coefficients. The project includes visualizations of the **learning curve**, the **3D plot of the cost function**, and other relevant equations. 

## Table of Contents 📚
- [Project Overview](#overview)
- [Problem Description](#problem-description)
- [Gradient Descent](#gradient-descent)
- [Cost Function (MSE)](#cost-function-mse)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Evaluation](#results-and-evaluation)
  - [Learning Curve](#learning-curve)
  - [3D Plot](#3d-plot)
  - [Equations](#equations)
- [Conclusion](#conclusion)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Problem Description 🔍

In **Multiple Linear Regression**, the goal is to model the relationship between multiple independent variables (features) and a dependent variable (target). This relationship is represented as:

\[
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n
\]

Where:
- \(y\) is the dependent variable (target).
- \(x_1, x_2, \dots, x_n\) are the independent variables (features).
- \(\beta_0, \beta_1, \dots, \beta_n\) are the regression coefficients to be determined.

The objective is to determine the optimal values for \(\beta_0, \beta_1, \dots, \beta_n\) that minimize the **Mean Squared Error (MSE)** cost function. 🎯

## Gradient Descent 🔽

**Gradient Descent** is an optimization algorithm used to minimize the **cost function** by iteratively adjusting the model parameters. The idea is to update the parameters in the direction of the steepest descent (negative gradient) of the cost function.

The update rule for the parameters \(\theta_j\) is:

\[
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
\]

Where:
- \(\theta_j\) is the \(j\)-th regression coefficient.
- \(\alpha\) is the **learning rate**.
- \(J(\theta)\) is the **cost function**.

This rule helps in reducing the cost over time until the model parameters converge to the optimal values. 🏁

## Cost Function (MSE) 📉

The **Mean Squared Error (MSE)** cost function measures the difference between the predicted and actual values:

\[
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2
\]

Where:
- \(m\) is the number of training examples.
- \(h_{\theta}(x^{(i)})\) is the predicted value for the \(i\)-th training example, calculated as:
  \[
  h_{\theta}(x^{(i)}) = \beta_0 + \beta_1 x_1^{(i)} + \beta_2 x_2^{(i)} + \dots + \beta_n x_n^{(i)}
  \]

The objective is to minimize \(J(\theta)\) using **Gradient Descent**, resulting in better model predictions. 🎯

## Installation ⚙️

### Clone the Repository

To get started, clone the repository to your local machine:

```bash
git clone https://github.com/Bushra-Butt-17/Gradient-Descent-Multiple-Linear-Regression.git
```

### Install Dependencies

Navigate to the project directory and install the required dependencies:

```bash
cd Gradient-Descent-Multiple-Linear-Regression
pip install -r requirements.txt
```

## Usage 🖥️

After setting up the project, you can run the **Gradient Descent** algorithm in various ways depending on the script you want to use:

### Option 1: Jupyter Notebooks 📓

1. **Multiple Linear Regression.ipynb**: This notebook provides a detailed walkthrough of the implementation of multiple linear regression using gradient descent.
2. **Regression_GD.ipynb**: This notebook offers another perspective or could be a more focused version of the algorithm with additional tests or exploration.

### Option 2: Python Script 🧑‍💻

To execute the gradient descent and model training using a Python script, run:

```bash
python public_tests.py
```

### Key Features 🌟:
- **Training the model**: The script will train a multiple linear regression model using gradient descent.
- **Visualizations**: After training, the following visualizations will be generated:
  1. **Learning curve** showing how the cost function decreases over time.
  2. **3D plot** of the cost function's surface.
  3. Display of key **equations** (Gradient Descent and MSE).

## Results and Evaluation 📊

### Learning Curve 📉

The learning curve tracks the change in the **cost function** over iterations of Gradient Descent. This demonstrates how the algorithm gradually converges towards the minimum cost. A typical learning curve should show a rapid decrease initially, followed by slower convergence as the algorithm nears the optimal solution.

![Learning Curve](images/learning_curve.png)

### 3D Plot 🌐

A **3D plot** is used to visualize the cost function's surface for two regression coefficients, illustrating how **Gradient Descent** converges to the optimal parameter values. It shows the path taken by the algorithm to find the minimum point.

![3D Plot](images/3d_plot.png)

### Equations 📐

The following are the key equations used in the project:

1. **Gradient Descent Update Rule**:
   \[
   \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
   \]

2. **Mean Squared Error (MSE) Cost Function**:
   \[
   J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2
   \]

These equations are essential to understanding how the model is trained using **Gradient Descent**.

![Equations](images/equations.png)

## Folder Structure 🗂️

The project follows the structure below:

```
├── data/                          # Data folder for storing datasets
├── Multiple Linear Regression.ipynb  # Jupyter notebook for implementing linear regression
├── public_tests.py                # Python script for public tests of the model
├── Regression_GD.ipynb            # Another Jupyter notebook for regression using GD
├── utils.py                       # Helper functions for gradient descent
└── Multiple Linear Regression.ipynb  # Duplicate or alternate version of the main notebook
```

## Conclusion 🎯

This project demonstrates the use of **Gradient Descent** to optimize a **Multiple Linear Regression** model. The visualizations, including the **learning curve** and **3D cost surface plot**, provide insight into the convergence of the algorithm. Through this project, you can see how **Gradient Descent** iteratively updates the regression coefficients to minimize the **MSE** and improve the model's performance.

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements 🙏

- This project demonstrates fundamental **machine learning** concepts, particularly for regression tasks.
- Special thanks to the **Python community** and libraries like **NumPy**, **Matplotlib**, and **Pandas** for their contributions to the scientific computing ecosystem.

---

