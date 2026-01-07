Model Card - Census Income Prediction
Model Details

    Developer: Yiquan Sun

    Date: January 2026

    Model Type: Random Forest Classifier

    Library: Scikit-learn via Python 3.8

Intended Use

    Primary Use Case: This model is intended to predict whether an individual's annual income exceeds $50K based on census demographic data.

    Target Users: Researchers or developers interested in socioeconomic data analysis.

    Out-of-Scope Use Cases: This model should not be used for making actual financial or credit-related decisions about individuals.

Training Data

    Source: UCI Machine Learning Repository (Census Income Dataset).

    Description: The dataset contains 32,561 records with features such as age, workclass, education, and marital status.

    Pre-processing: * Leading and trailing spaces were removed from strings to ensure data consistency.

        Categorical variables were encoded using OneHotEncoder.

        The target label was transformed using LabelBinarizer.

Evaluation Data

    Description: The model was evaluated on a 20% hold-out test set derived from the original census data.

    Metrics: Evaluation focused on Precision, Recall, and F1-Score to ensure a balance between identifying high-income earners and minimizing false positives.

Metrics

    Precision: 0.95

    Recall: 0.90

    F1-Score: 0.92

Slicing Analysis

    Methodology: Performance was evaluated across categorical slices (e.g., Education, Race, Sex) to identify potential bias.

    Findings: The model generally performs best on the Bachelors and Masters education slices, while showing lower recall on the Private workclass slice. Full results are documented in slice_output.txt.

Ethical Considerations

    Bias: The dataset is from 1994 and may reflect historical socioeconomic biases regarding gender and race.

    Privacy: The data is anonymized, but model predictions based on sensitive attributes should be handled with care to avoid discriminatory outcomes.

Caveats and Recommendations

    Recency: Given the data is from 1994, the income thresholds and job categories may not reflect the modern economy.

    Update Strategy: If used in production today, the model should be retrained on more recent census data to account for inflation and shifting labor trends.