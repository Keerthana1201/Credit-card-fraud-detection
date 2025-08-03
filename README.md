**Credit Card Fraud Detection using Logistic Regression**

This project addresses the problem of detecting fraudulent credit card transactions using a supervised machine learning approach. It demonstrates a complete workflow involving data preprocessing, feature scaling, class imbalance handling using SMOTE, model training using logistic regression, and performance evaluation.

**Dataset**
The dataset used for this project is publicly available on Kaggle and contains transactions made by European cardholders over two days in September 2013. It includes 284,807 transactions, out of which only 492 are fraudulent, making it a highly imbalanced classification problem.

The features are the result of a Principal Component Analysis (PCA) transformation applied to the original data for confidentiality. Features V1 through V28 represent these components, while 'Amount' and 'Time' are included without anonymization. The target variable 'Class' indicates whether a transaction is fraudulent (1) or not (0).

**Objective**
The goal is to develop a machine learning model that can accurately detect fraudulent transactions, particularly focusing on minimizing false negatives (fraudulent transactions that are not detected), while also avoiding a high number of false positives (non-fraudulent transactions incorrectly flagged as fraud).

**Methodology**
The pipeline begins by scaling the numerical features 'Amount' and 'Time' using StandardScaler, which ensures that these features are on the same scale as the PCA components. To address the significant class imbalance in the dataset, SMOTE (Synthetic Minority Over-sampling Technique) is applied. This technique generates synthetic samples for the minority class, enabling the model to learn from both classes more effectively.

After resampling, a logistic regression model is trained on the balanced dataset. Logistic regression is chosen as a baseline model due to its simplicity, interpretability, and effectiveness for binary classification tasks.

**Model Performance and Analysis**
Following model training, evaluation is performed using a separate test set. The confusion matrix shows that the model correctly identifies a large majority of both fraudulent and non-fraudulent transactions. Specifically, out of over 85,000 actual fraud cases in the test set, the model correctly identified more than 78,000. Similarly, among the non-fraudulent transactions, over 83,000 were correctly classified.

The precision for fraudulent transactions is approximately 97%, indicating that when the model predicts a transaction as fraudulent, it is correct most of the time. This is important for reducing the number of false alarms, which can negatively impact user trust and create unnecessary manual reviews. The recall for the fraud class is about 92%, suggesting the model is able to successfully detect the majority of fraudulent transactions.

The model achieved an F1-score of approximately 95% for both classes, showing a strong balance between precision and recall. Overall accuracy is 95%, but more importantly, the macro and weighted averages of the F1-score also reflect a consistently strong performance across both classes.

These results demonstrate that even a relatively simple model like logistic regression can be highly effective in detecting fraud when the data is properly preprocessed and balanced. The use of SMOTE was critical in achieving this performance by ensuring that the model was exposed to sufficient examples of fraudulent transactions during training.

**Conclusion**
This project shows that with the right data preparation techniques, logistic regression can serve as a strong baseline for fraud detection tasks. Despite the severe class imbalance in the original dataset, the use of SMOTE significantly improved the model’s ability to generalize and detect fraudulent behavior. High precision minimizes the number of false positives, while high recall ensures that most fraudulent transactions are caught.
