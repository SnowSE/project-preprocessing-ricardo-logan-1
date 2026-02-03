[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/RgEqoZ3m)
[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=22410642)
# Class Project: Preprocessing and Exploratory Data Analysis (EDA)
__MATH 3480__: Machine Learning

## Project Overview
A survey was completed to examine the mental health of teenagers and young adults. The goal of this project is to perform comprehensive data preprocessing on the *young-people-survey-responses* dataset to prepare it for predicting loneliness in young people.

**Submission Format:** Submit a **written report** (PDF or Word document) that includes all deliverables, code snippets, figures with captions, and documentation of your preprocessing decisions.

---

## Report Structure and Requirements

Your report should include:
- **Code snippets**: Relevant code illustrating key steps (not your entire notebook)
- **Figures and visualizations**: Charts, graphs, and tables at readable sizes
- **Figure captions**: Descriptive captions for every figure
- **Explanations**: Clear documentation of steps and justifications for decisions
- **Results**: Outcomes of each preprocessing step with statistics or summaries

---

## Part 1: Data Loading and Initial Exploration

### Deliverables:

1. **Dataset Overview**
   - Report dimensions (rows × columns)
   - Table showing first 5-10 rows
   - Summary of data types

2. **Initial Data Quality**
   - Identify the target variable (what we are trying to predict)
   - Summary table of missing values per column

***Suggested* Figures:** Sample data table, target distribution bar chart

---

## Part 2: Exploratory Data Analysis (EDA)

**Important Note:** EDA and preprocessing are iterative processes done side-by-side. You need EDA to understand what preprocessing is required, and you need preprocessing to properly analyze some variables. You may find yourself moving back and forth between Parts 2 and 3.

### Deliverables:

1. **Univariate Analysis**
   - For **numerical variables**: Include histograms or box plots for key variables
   - For **categorical variables**: Include bar charts showing frequency distributions for key variables
   - Caption each figure noting important observations (skewness, outliers, imbalanced categories, etc.)
   - Identify which variables have distributions that may require special handling

2. **Relationship to Target Variable (Bivariate Analysis)**
   - Create visualizations exploring the relationship between important variables and the target (loneliness)
   - For **numerical predictors**: Use box plots, violin plots, or scatter plots grouped by target
   - For **categorical predictors**: Use grouped bar charts or stacked bar charts
   - Caption each figure explaining what relationship (if any) you observe

3. **Correlation Analysis**
   - Create a correlation heatmap for numerical variables
   - Identify highly correlated variable pairs (correlation > 0.7 or < -0.7)
   - Discuss potential multicollinearity concerns
   - Note which variables appear most correlated with the target

4. **Key Findings Summary**
   - Write 1-2 paragraphs summarizing your most important findings:
     - Which variables appear most predictive of loneliness?
     - Which variables show little to no relationship with the target?
     - Are there any surprising relationships?
     - What preprocessing concerns did you identify (outliers, skewness, etc.)?

**Suggested Figures:** 
- Histograms/box plots of key numerical variables
- Bar charts of key categorical variables
- Box plots or violin plots showing variable distributions by target class
- Correlation heatmap
- Scatter plots for interesting relationships

**Note:** This EDA will inform your preprocessing decisions in Part 3. For example:
- High correlation between variables may lead you to remove redundant features
- Severely skewed distributions may affect your imputation method choice
- Variables showing no relationship to the target may be candidates for removal

---

## Part 3: Data Preprocessing

### 3.1 Feature Selection

### Deliverables:

1. **Variable Removal Documentation**
   - Table listing removed variables with justifications
   - **Reference your EDA findings** when explaining removals

| Variable Name | Reason for Removal |
|---------------|-------------------|
| respondent_id | Unique identifier, no predictive value |
| variable_x | No correlation with target (see EDA correlation heatmap) |

2. **Feature Count Summary**
   - Starting vs. remaining feature count

---

### 3.2 Handling Missing Values

### Deliverables:

1. **Missing Data Analysis**
   - Visualization of missing value percentages per column
   - Summary of missing values per row

2. **Removal Documentation**
   - Number of columns removed (>10% missing)
   - Number of rows removed (>10% missing)
   - Before/after dataset dimensions

3. **Imputation Strategy Table**

| Variable Name | Variable Type | % Missing | Imputation Method | Justification |
|---------------|---------------|-----------|-------------------|---------------|
| age | Numerical | 3.2% | Median | Data showed right skew in EDA |

4. **Imputation Results**
   - Before/after comparison of missing values

**Suggested Figures:** Missing value bar chart, before/after comparison, distribution plots of imputed values

---

### 3.3 Encoding Categorical Variables

### Deliverables:

1. **Categorical Variable Identification**
   - List all categorical variables
   - Classify each as nominal or ordinal

2. **Encoding Strategy Table**

| Variable Name | Variable Type | Encoding Method | Justification | Resulting Columns |
|---------------|---------------|-----------------|---------------|-------------------|
| gender | Nominal | One-Hot | No inherent order | 3 columns |

3. **Encoding Results**
   - Feature count before and after encoding
   - Example of transformed variable (small sample)

**Suggested Figures:** Before/after encoding example, feature count comparison chart

---

## Part 4: Cross Validation (Train-Test Split)

### Deliverables:

1. **Split Strategy**
   - State train-test ratio and random_state value
   - Brief justification

2. **Dataset Dimensions Table**

| Dataset | Samples | Features | % of Total |
|---------|---------|----------|------------|
| Training | 850 | 45 | 70% |
| Testing | 365 | 45 | 30% |

3. **Target Distribution Analysis**
   - Side-by-side visualizations of target distribution in train vs. test sets
   - Caption confirming balanced split

**Suggested Figures:** Side-by-side bar charts, dimension summary table

---

## Part 5: Feature Scaling

### Deliverables:

1. **Scaling Decision Table**

| Feature Name | Data Type | Scaling Applied? | Justification |
|--------------|-----------|------------------|---------------|
| age | Continuous | Yes | Wide range observed in EDA |
| gender_male | Binary | No | Already 0/1 |

2. **Scaling Methodology**
   - Confirm scaler fit on training data only
   - Explain why (prevents data leakage)

3. **Scaling Verification**
   - Summary statistics showing mean ≈ 0, std ≈ 1 for scaled features

4. **Visual Comparison**
   - Distribution plots for 2-3 features before/after scaling

**Suggested Figures:** Before/after box plots, summary statistics table

---

## Part 6: Validation (Optional - Not Graded)

If included: Confirm multiple linear regression model ran without errors.

---

## Part 7: Summary and Reflection

### Deliverables:

1. **Processing Summary Table**

| Preprocessing Step | Initial | Final | Change |
|-------------------|---------|-------|--------|
| Samples | 1,500 | 1,215 | -285 |
| Features | 150 | 45 | -105 |
| Missing Values | 8,450 | 0 | All handled |

2. **Key Insights**
   - Most important findings from EDA
   - Most challenging aspect of preprocessing
   - Variables with most missing data
   - New features from one-hot encoding
   - Any concerns about final dataset
   - How EDA informed your preprocessing decisions

---

## Report Formatting Guidelines

- **Length**: 5 is ideal, but can be 3-7 pages (including figures)
- **Font**: 11-12 point, readable font
- **Margins**: 1-inch on all sides
- **File format**: PDF or .docx
- **Figures**: Readable size with numbered captions, does not take up excessive space
- **Code**: Only relevant snippets, properly formatted
- **Writing**: Clear, professional, concise

---

## Submission Checklist

- [ ] All sections included with deliverables
- [ ] EDA section includes sufficient visualizations and analysis
- [ ] EDA findings referenced in preprocessing decisions
- [ ] All figures captioned and referenced in text
- [ ] All tables clearly formatted
- [ ] Justifications for all major decisions
- [ ] Well-organized with clear headings
- [ ] Proofread for errors
- [ ] File named: LastName_FirstName_Preprocessing_Project.pdf

---

# Grading Rubric

| Category | Criteria | Points |
|----------|----------|--------|
| **1. Data Loading & Exploration** | Dataset overview with dimensions, sample, data types                 | 1 |
|                                   | Initial data quality assessment with figures                         | 1 |
| **2. Exploratory Data Analysis**  | Univariate analysis with appropriate visualizations and observations | 2 |
|                                   | Relationship to target analysis with clear visualizations            | 2 |
|                                   | Correlation analysis with heatmap and interpretation                 | 1.5 |
|                                   | Key findings summary showing thoughtful analysis                     | 1.5 |
| **3. Preprocessing**              | Discussion of Feature Selection (removed variables)                  | 2 |
|                                   | Missing value analysis with visualizations                           | 2 |
|                                   | Discussion of imputation strategy                                    | 2 |
|                                   | All categorical variables identified and classified                  | 1 |
|                                   | Clear justifications for encoding choices                            | 2 |
| **4. Train-Test Split**           | Split strategy explained with appropriate ratio                      | 3 |
| **5. Feature Scaling**            | Scaling decision table with justifications                           | 1 |
|                                   | Correct methodology (fit on train, no leakage)                       | 2 |
| **6. Report Quality**             | Well-organized, professionally formatted;<br>Figures/tables with captions, properly sized;<br>Clear writing, free of major errors;<br>Summary with thoughtful insights connecting EDA to preprocessing | 3 |
| **7. Code**                       | Code file is included and works without errors                       | 3 |
| | | |
| **Total:** | | 30 |

-----

This assignment was created with the assistance of AI.