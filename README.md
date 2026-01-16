# SMS Spam Detection Using TF-IDF + Naive Bayes (Machine Learning)

This project builds a **text classification** model to detect whether an SMS message is **Spam** or **Ham (Not Spam)**.
It uses **TF-IDF** for text feature extraction and **Multinomial Naive Bayes** for classification. The project follows the standard ML workflow:

**Problem Statement → Selection of Data → Collection of Data → EDA → Train/Test Split → Model Selection → Evaluation Metrics**

---

## Problem Statement
Spam messages waste time and can be harmful (scams, phishing).  
The goal of this project is to classify SMS messages into:
- **Ham (0)**: normal message
- **Spam (1)**: unwanted/promotional/scam message

**Output:** Predicted label for each message (Spam/Ham)

---

## Selection of Data
**Dataset Type Used:** Text dataset (SMS messages) with binary labels

Why this dataset is suitable:
- Real-world NLP classification problem
- Clear labels (spam vs ham)
- Great for practicing text preprocessing + ML modeling

---

## Collection of Data
The dataset is uploaded in Google Colab using:
- `from google.colab import files`
- `files.upload()`

Then loaded using:
- `pd.read_csv("spam.csv")`

---

## EDA (Exploratory Data Analysis)
EDA is kept simple for beginners:
- Load the dataset and verify columns (`label`, `message`)
- Check label mapping works correctly (ham → 0, spam → 1)
- Confirm data splits properly before training

---

## Dividing Training and Testing
The dataset is split using `train_test_split`:
- Train set: used to learn from text
- Test set: used to evaluate performance

Used in code:
- `test_size=0.2` (80% train, 20% test)
- `random_state=42` (reproducible)

---

## Data Preprocessing (Text to Numbers)
ML models cannot understand raw text, so we convert text into numeric features using **TF-IDF**.

**TF-IDF step used:**
- `TfidfVectorizer(stop_words='english')`
- `fit_transform` on training messages
- `transform` on test messages

Why TF-IDF:
- Gives importance to meaningful words
- Reduces the effect of very common words
- Works well for spam detection

---

## Model Selection
**Model used:** Multinomial Naive Bayes (`MultinomialNB`)

Why Naive Bayes:
- Very effective for text classification
- Fast training and good accuracy with TF-IDF features
- Works well with word frequency-based inputs

---

## Evaluation Metrics (Used in this Project)
This project evaluates the model using:
- **Accuracy Score:** overall correctness
- **Classification Report:** precision, recall, and F1-score (Spam vs Ham)

Used in code:
- `accuracy_score(y_test, y_pred)`
- `classification_report(y_test, y_pred)`

---

## Custom Prediction (Inference)
A custom function is created to predict a new message:

- Converts input text using the trained TF-IDF vectorizer
- Predicts using Naive Bayes model
- Returns **"Spam"** or **"Ham"**

Example tests in code:
- `"You won a free prize"` → Spam
- `"Let's meet tomorrow"` → Ham

---

## Main Libraries Used (and why)

1. `pandas`  
   - Loads CSV data and manages it as a DataFrame.

2. `sklearn.model_selection.train_test_split`  
   - Splits the dataset into training and testing sets.

3. `sklearn.feature_extraction.text.TfidfVectorizer`  
   - Converts text messages into TF-IDF numeric vectors.

4. `sklearn.naive_bayes.MultinomialNB`  
   - Trains the spam/ham text classifier.

5. `sklearn.metrics`  
   - Evaluates performance using accuracy and classification report.

6. `google.colab.files`  
   - Uploads dataset file into Colab environment.

---

## Output
- Printed model performance:
  - Accuracy Score
  - Classification Report
- Custom message predictions:
  - Spam/Ham result for user input text

---

## Developer
Grishma C.D
