# Employee Salary Prediction

This repository contains a machine learning project that predicts whether an employee's annual salary is greater than $50,000 or less than or equal to $50,000. The prediction is based on various demographic and employment-related features.

## Table of Contents

* [Project Overview](https://www.google.com/search?q=%23project-overview)

* [Dataset](https://www.google.com/search?q=%23dataset)

* [Methodology](https://www.google.com/search?q=%23methodology)

* [Results](https://www.google.com/search?q=%23results)

* [Streamlit Application](https://www.google.com/search?q=%23streamlit-application)

* [Installation and Usage](https://www.google.com/search?q=%23installation-and-usage)

* [Contributing](https://www.google.com/search?q=%23contributing)

* [License](https://www.google.com/search?q=%23license)

## Project Overview

The goal of this project is to build and deploy a machine learning model that can accurately classify an individual's income bracket based on publicly available census data. This can be useful for various applications, such as targeted advertising, demographic analysis, or understanding factors influencing income.

## Dataset

The dataset used in this project is sourced from the UCI Adult Income Dataset. It contains 48,842 rows and 15 columns, with features such as:

* `age`: Age of the individual.

* `workclass`: Type of employer (e.g., Private, Self-emp-not-inc, Local-gov).

* `fnlwgt`: Final weight (statistical sampling weight).

* `education`: Highest level of education achieved.

* `educational-num`: Numerical representation of education.

* `marital-status`: Marital status.

* `occupation`: Type of occupation.

* `relationship`: Relationship status.

* `race`: Race of the individual.

* `gender`: Gender of the individual.

* `capital-gain`: Capital gains.

* `capital-loss`: Capital losses.

* `hours-per-week`: Number of hours worked per week.

* `native-country`: Country of origin.

* `income`: Target variable, either `<=50K` or `>50K`.

### Data Cleaning and Preprocessing

The following steps were performed to clean and preprocess the data:

* **Handling Missing Values**: Replaced '?' values in `workclass` and `occupation` with 'Not Listed' and 'Others' respectively.

* **Outlier Treatment**: Removed outliers from `age`, `capital-gain`, `capital-loss`, and `hours-per-week` columns based on boxplot analysis.

* **Feature Removal**: Dropped `educational-num` (redundant with `education`) and `fnlwgt` (statistical weight, not a direct predictor of income).

* **Label Encoding**: Converted all categorical features into numerical representations using `LabelEncoder`.

## Methodology

The project follows a standard machine learning pipeline:

1. **Data Loading and Initial Exploration**: Loaded the dataset and performed initial checks for shape, head, tail, and missing values.

2. **Data Preprocessing**: Applied the cleaning and encoding steps mentioned above.

3. **Model Training and Evaluation**:

   * The data was split into training and testing sets (80% train, 20% test).

   * A `StandardScaler` was used within a `Pipeline` to scale numerical features.

   * Several classification models were trained and evaluated:

     * Logistic Regression

     * Random Forest Classifier

     * Gradient Boosting Classifier

     * K-Nearest Neighbors Classifier

     * Support Vector Machine (SVC)

   * Each model's accuracy and classification report (precision, recall, f1-score) were calculated.

4. **Model Selection**: The model with the highest accuracy was selected as the best model.

## Results

After training and evaluating various models, the **Gradient Boosting Classifier** achieved the highest accuracy:

* **Gradient Boosting Accuracy**: 0.8640 (86.40%)

Here's a summary of the accuracies:

| **Model** | **Accuracy** | 
| Logistic Regression | 0.8007 | 
| Random Forest | 0.8485 | 
| Gradient Boosting | 0.8640 | 
| K-Nearest Neighbors | 0.8284 | 
| Support Vector Machine | 0.8342 | 

The best model (`best_model.pkl`) is saved for later use in the Streamlit application.

## Streamlit Application

A Streamlit web application (`app2.py`) is provided to allow interactive predictions. Users can input various employee details through a sidebar, and the application will output the predicted salary bracket.

You can access the live Streamlit application here: <https://74366ee085ce.ngrok-free.app>

### Features in the Streamlit App:

* **Age**: Slider (18-75)

* **Workclass**: Selectbox with cleaned categories.

* **Education**: Selectbox with cleaned categories.

* **Marital Status**: Selectbox with relevant categories.

* **Occupation**: Selectbox with cleaned categories.

* **Relationship**: Selectbox with relevant categories.

* **Race**: Selectbox with relevant categories.

* **Gender**: Selectbox (Male/Female).

* **Capital Gain**: Number input.

* **Capital Loss**: Number input.

* **Hours per week**: Slider (1-99).

* **Native Country**: Selectbox with relevant categories.

## Installation and Usage

To run this project locally, follow these steps:

### Prerequisites

* Python 3.8+

* `pip` (Python package installer)

### 1. Clone the repository:

```
git clone <repository_url>
cd employee-salary-prediction

```

### 2. Create a virtual environment (recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

```

### 3. Install dependencies:

```
pip install -r requirements.txt

```

*(Note: You might need to create a `requirements.txt` file by running `pip freeze > requirements.txt` after installing all necessary libraries in your environment, or manually add `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `streamlit`, `pyngrok`.)*

### 4. Run the Jupyter Notebook:

Open and run the `employee_salary_prediction_ml.ipynb` notebook to train the model and generate `best_model.pkl`.

```
jupyter notebook employee_salary_prediction_ml.ipynb

```

### 5. Run the Streamlit application:

Ensure you have `app2.py` and `best_model.pkl` in the same directory.

```
streamlit run app2.py

```

This will open the Streamlit application in your web browser.

### Running with Ngrok (for public access)

If you want to expose your Streamlit app to the internet (e.g., for demonstration purposes), you can use `ngrok`.

1. **Install ngrok**:

   ```
   pip install pyngrok
   
   ```

2. **Get your ngrok authtoken**:
   Sign up at [ngrok.com](https://ngrok.com/) and get your authtoken from the dashboard.

3. **Authenticate ngrok**:

   ```
   ngrok authtoken YOUR_NGROK_AUTHTOKEN
   
   ```

   (Replace `YOUR_NGROK_AUTHTOKEN` with your actual token.)

4. **Run the Streamlit app and ngrok tunnel (as shown in the notebook):**
   The notebook already contains the code to run Streamlit and ngrok. Just execute the last few cells. You will get a public URL in the output.

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

