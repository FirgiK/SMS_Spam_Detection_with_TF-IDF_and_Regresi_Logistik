# SMS Spam Detection Model (TF-IDF + Logistic Regression)

This is a machine learning project for text classification that aims to build an accurate SMS spam detection model.

This model uses a classic machine learning approach (`TfidfVectorizer` + `LogisticRegression` from Scikit-learn) on the "SMS Spam Collection" dataset from UCI.

**Final Test Accuracy: ~98%** (with successful generalization on new data)

---

## 1. Dataset Explanation

### 1.1. The Case Study
The task is **SMS Spam Detection**, a binary text classification problem. The goal is to train a model that can accurately differentiate between legitimate SMS (ham) and unsolicited messages (spam) based on their text content.

### 1.2. Column Explanations
We are using the public "SMS Spam Collection" dataset from UCI, available on Kaggle as `spam.csv`.

The `spam.csv` file has 5 columns, but we will only focus on the first two:
* **v1 (label_text)**: The category label, containing "ham" or "spam". This will be our target (`y`).
* **v2 (text)**: The raw text of the SMS message. This will be our feature (`X`).

---

## 2. Data Understanding

The dataset consists of 5,169 unique rows after cleaning nulls and duplicates. The class distribution is highly **imbalanced**, with approximately **87% Ham** and **13% Spam**.

This is a critical insight, as a naive model could achieve 87% accuracy by simply guessing "Ham" every time. Therefore, techniques like `class_weight='balanced'` are necessary during training.

#### Word Clouds (Ham vs. Spam)
Word cloud visualizations show a clear vocabulary difference between the two classes, even after preprocessing.

*(You can insert your `wordcloud_ham.png` and `wordcloud_spam.png` images here)*

---

## 3. Data Text Processing

The text processing pipeline is as follows:

1.  **Text Normalization**: All text is converted to lowercase.
2.  **Stopword & Punctuation Removal**: Common words (e.g., "the", "is", "in") and all punctuation are removed.
3.  **Tokenization**: Sentences are split into individual words (tokens).
4.  **Lemmatization**: Words are converted to their root form (e.g., "running" -> "run").
5.  **Data Splitting**: The data is split into 80% Train and 20% Test sets, using `stratify=y` to maintain the 87/13 class distribution in both sets.
6.  **Text Vectorization**: `TfidfVectorizer` is used to convert the clean text into a numerical matrix. TF-IDF weights words based on their importance, not just their frequency.

---

## 4. Data Modeling

### Model Explanation
We are using **Logistic Regression**.
* **Why not Deep Learning (GRU/LSTM)?** Our initial attempts with GRU models on other datasets resulted in extreme overfitting and a complete failure to generalize to new data.
* **Why Logistic Regression?** It is a fast, efficient, and highly stable model for text classification. It is less prone to overfitting and provides a very strong performance baseline.
* **Key Improvement:** We use the `class_weight='balanced'` parameter. This is critical for handling the imbalanced dataset. It tells the model to "punish" itself more severely for misclassifying the minority class (Spam), making it more vigilant.

---

## 5. Data Evaluation

The final model achieves a **Test Accuracy of ~98%**.

The Classification Report shows excellent performance, notably a **Recall of 0.96 for Spam**. This means the model successfully found 96% of all spam messages in the test set. This is arguably the most important metric, as we would rather have 1-2 ham messages go to the spam folder than let 1 spam message get into the inbox.

*(You can insert your classification report and confusion matrix images here)*

---

## 6. Data Prediction (The Final Test)

The trained model was tested on 4 new, unseen messages. It succeeded where all previous models failed:

| Message | Prediction |
| :--- | :--- |
| "Congratulations! You've won a $1,000..." | **Spam (1)** |
| "Hi team, just a reminder about our weekly meeting..." | **Ham (0)** |
| "URGENT: Your account has been suspended..." | **Spam (1)** |
| "Thanks for your order #12345! Your package..." | **Ham (0)** |

This confirms the model has learned the *concept* of spam and can successfully generalize.

---

## Project Journey & Debugging Summary

This project is a case study in why dataset quality is more important than model complexity.

1.  **Initial Model (GRU + Synthetic Data):** The first model was trained on the `phishing_legit...` dataset and achieved 100% accuracy. This was a **false positive** caused by **Data Leakage**. The model simply learned to find the word "keywords:" in the synthetic spam emails and failed to predict real-world spam.

2.  **Second Model (GRU + Enron Data):** The second model used the realistic `enron_spam_data.csv` dataset. It achieved 99.9% accuracy but **failed again** on new data. This was **Contextual Overfitting** it perfectly memorized "Enron spam" but had no vocabulary for "general spam" (e.g., "prize", "congratulations").

3.  **Final Model (TF-IDF + SMS Data):** The final, successful model was trained on the clean, verified, and realistic `spam.csv` (UCI) dataset. We also switched to a simpler, more robust model (Logistic Regression) with `class_weight='balanced'`.

**Key Takeaway:** A simple model on a realistic, high-quality dataset will always outperform a complex model on a flawed or synthetic dataset.
