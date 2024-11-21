
# Boston Housing Price Prediction - Machine Learning Project

## 🚀 **Project Overview**

This project is a part of **Udacity's Machine Learning Nanodegree** program and involves the construction of a machine learning model to predict the **median value of homes** in Boston. The dataset includes features like the number of rooms, percentage of lower-status residents, and the pupil-teacher ratio. The model uses the **Decision Tree Regressor** to make accurate price predictions for real estate.

### **What You Will Learn:**
- Data exploration and visualization
- Building machine learning models with **Decision Trees**
- Hyperparameter tuning using **GridSearchCV**
- Evaluating model performance with metrics like **R²**
- Deploying the model for future predictions

---

## 💻 **Key Features of This Project**

- **Data Exploration**: Analyze the features and inspect their correlation with home prices.
- **Model Training**: Using **DecisionTreeRegressor** for predictions.
- **Hyperparameter Tuning**: Optimize the model using **GridSearchCV**.
- **Performance Evaluation**: Measure the model’s performance with **R² score** and visualize results.
- **Future Predictions**: Predict prices for new clients using trained models.

---

## 📂 **Project Structure**

Here’s a breakdown of the project files and their purposes:

```
boston_housing/
├── boston_housing.ipynb          # Jupyter notebook for model development
├── housing.csv                  # Boston Housing dataset (489 rows of data)
├── visuals.py                   # Helper functions for visualizations (graphs, plots)
├── README.md                    # Project documentation (you're here!)
```

---

## 🧰 **Installation & Setup**

To get started with this project, you’ll need the following Python libraries:

- **Python** (version 3.x or higher)
- **NumPy**: For numerical computations
- **Pandas**: For data manipulation and analysis
- **Matplotlib**: For visualizations
- **Scikit-learn**: For machine learning models and evaluation metrics

### Step 1: Install Python

If you don’t have Python installed, download and install [Anaconda](https://www.anaconda.com/products/individual) (which comes with most libraries pre-installed).

### Step 2: Install Required Libraries

You can install the necessary Python libraries by running the following command in your terminal:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## 📊 **Data Overview**

This dataset, from the **UCI Machine Learning Repository**, consists of 489 data points, each describing a property in Boston with 3 features:

- **RM**: Average number of rooms per dwelling
- **LSTAT**: Percentage of the population considered lower status
- **PTRATIO**: Pupil-teacher ratio by town

The target variable we are predicting is:

- **MEDV**: Median value of homes (in thousands of dollars)

### Example Data:

| RM   | LSTAT | PTRATIO | MEDV   |
|------|-------|---------|--------|
| 6.6  | 11.5  | 15.3    | 24.0   |
| 5.0  | 12.4  | 17.0    | 19.0   |
| 8.0  | 8.6   | 20.0    | 28.5   |

---

## 🧠 **Model Development**

We use a **DecisionTreeRegressor** to predict house prices based on the input features. The model is fine-tuned with **GridSearchCV** to optimize the **max_depth** parameter for better accuracy.

### Steps:
1. Load the dataset and clean it.
2. Split the dataset into training and testing sets.
3. Build and train the decision tree regressor.
4. Apply **GridSearchCV** to tune the hyperparameters.
5. Evaluate the model's performance using the **R²** score.

---

## 🔧 **Model Evaluation**

The performance of the model is evaluated based on the **R²** score, which tells us how well our model is predicting the prices. A high R² score (close to 1) indicates good predictive power.

**Model Performance Example**:
```python
score = performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
print(f"Model has a coefficient of determination, R², of {score:.3f}.")
```

---


## 🚀 **How to Run**

1. Clone or download this repository.
2. Open a terminal or command prompt in the project directory.
3. Launch the Jupyter Notebook:
   ```bash
   jupyter notebook boston_housing.ipynb
   ```
4. Run the notebook cells to train the model and make predictions.

---

## 🔍 **How to Contribute**

This project is part of Udacity's **Machine Learning Nanodegree**. However, feel free to fork it, make improvements, or use it as a reference for your own projects.

---

### **Acknowledgments**
- **Udacity** for the **Machine Learning Nanodegree** and providing the foundational structure for this project.
- **UCI Machine Learning Repository** for the original dataset.

