# Principles of Data Science Capstone

## Author: Ali Lugo  
## Date: December 9, 2024  

### **Project Overview**
This capstone project for the **PODS Data Science Course** investigates various factors influencing professor ratings on **RateMyProfessors.com**. Using statistical analysis and machine learning models, this project explores how professor gender, experience, difficulty level, online teaching, student willingness to retake classes, and perceived attractiveness impact ratings.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Key Analyses](#key-analyses)
- [Models](#models)
- [Findings](#findings)
- [Installation & Requirements](#installation--requirements)
- [How to Run](#how-to-run)
- [Conclusion](#conclusion)

---

## **Datasets**
The project utilizes two datasets:
1. **Numerical Dataset (`rmpCapstoneNum.csv`)**: Includes average rating, difficulty level, number of ratings, and gender distribution.
2. **Qualitative Dataset (`rmpCapstoneQual.csv`)**: Includes major/field of study, university, and state.

The datasets were cleaned and merged to form a comprehensive dataset for analysis.

---

## **Methodology**
### **Data Cleaning & Preprocessing**
- Renamed columns for clarity.
- Removed rows with missing values in **Average Rating**.
- Applied a reliability threshold (min. 5 ratings per professor) to filter out unreliable data.
- Merged numerical and qualitative datasets.

### **Statistical Analysis & Hypothesis Testing**
- Independent **t-tests** were used to compare groups.
- **Correlation Analysis** to examine relationships between features.
- **Multiple Linear Regression** to predict professor ratings.
- **Logistic Regression** to classify whether a professor receives a 'pepper' icon (indicating attractiveness).

---

## **Key Analyses**
1. **Gender Bias**: Do male professors receive higher ratings than female professors?
2. **Experience Impact**: Does more experience (measured by number of ratings) correlate with higher ratings?
3. **Difficulty vs. Ratings**: Do harder classes result in lower professor ratings?
4. **Online Teaching Effect**: Do professors who teach more online classes receive different ratings than in-person professors?
5. **Retake Likelihood**: How does the proportion of students willing to retake a professor’s class relate to their rating?
6. **Attractiveness Bias**: Do 'hot' professors (indicated by a pepper icon) receive higher ratings?
7. **Multivariable Regression**: Which factors best predict professor ratings?
8. **Classification Model**: Can we predict whether a professor receives a 'pepper' icon based on multiple factors?
9. **Exploratory Analysis**: How do professor ratings vary by field of study?

---

## **Models**
1. **Simple Linear Regression**
   - Predicts **Average Rating** using a single factor (e.g., Difficulty, Number of Ratings).
2. **Multiple Linear Regression**
   - Predicts **Average Rating** using all available factors.
   - Evaluates feature importance and collinearity.
3. **Logistic Regression**
   - Classifies whether a professor receives a **pepper icon** based on different feature sets.
   - Addressed **class imbalance** using upsampling.

---

## **Findings**
### **Key Results**
- **Gender Bias:** Male professors received **statistically significantly** higher ratings than female professors, though the effect size was small.
- **Experience Effect:** Number of ratings had a **statistically significant** but **negligible** impact on ratings.
- **Difficulty vs. Ratings:** A **strong negative correlation (-0.62)** was found between difficulty and ratings.
- **Online Teaching Impact:** Professors teaching many online classes received significantly lower ratings.
- **Retake Likelihood:** A **strong positive correlation (0.88)** was found between retake likelihood and average ratings.
- **Attractiveness Bias:** 'Hot' professors received significantly higher ratings.
- **Regression Model Performance**
  - Single-factor models explained **38.3% (R²)** of rating variance.
  - A **multi-factor model improved R² to 79.8%**, indicating that multiple features together better predict ratings.
- **Classification Model Performance**
  - **AU(ROC) for predicting 'pepper' icon:**
    - **Using only ratings:** 0.78
    - **Using all factors:** 0.79 (minor improvement)

---

## **Installation & Requirements**
### **Dependencies**
Ensure you have the following Python libraries installed:
```sh
pip install numpy pandas scipy statsmodels seaborn matplotlib scikit-learn
```
---
## **How to Run**
1. Clone Repository & Navigate to Project Directory
```sh
git clone <repository_url>
cd PODSCapstone
```
2. Run the Python Script
```sh
python PODSCapstone.py
```
3. Expected Outputs
- Summary statistics and t-test results.
- Visualizations (Boxplots, Scatterplots, Regression Lines, ROC Curves, Feature Importance).
- Model performance metrics (R², RMSE, AU(ROC), classification reports).



---
## **Conclusion**
This project highlights how student evaluations of professors are influenced by multiple factors, including gender, perceived difficulty, and attractiveness. While statistical significance was found in various analyses, practical significance varied. The findings suggest that student ratings may be biased and influenced by non-teaching-related attributes. This analysis can be useful in discussions about the reliability of student evaluations in academic settings.

### **Future Work**
Incorporating textual analysis of qualitative reviews to extract sentiment and themes.
Testing alternative machine learning models (e.g., decision trees, neural networks) for prediction.
Expanding dataset to include more diverse institutions for generalizability.

