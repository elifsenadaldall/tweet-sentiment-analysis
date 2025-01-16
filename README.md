# Sentiment Analysis on Tweets Using Logistic Regression, LSTM, and BERT

## Author: Elif Sena Daldal

---

## **Abstract**
This project analyzes tweets to classify sentiments into positive, neutral, or negative categories using three models:
1. Logistic Regression
2. Long Short-Term Memory (LSTM)
3. Bidirectional Encoder Representations from Transformers (BERT)

The analysis leverages the Sentiment140 dataset and incorporates preprocessing techniques such as text cleaning, tokenization, and feature engineering. Performance is evaluated using metrics like Accuracy, Precision, Recall, and F1-Score, highlighting the strengths of both traditional machine learning and advanced NLP methodologies for sentiment analysis.

---

## **Introduction**
Sentiment analysis is a key task in natural language processing (NLP) with applications in:
- Understanding public opinion
- Analyzing customer feedback
- Tracking social media trends

### **Objectives**
1. Implement and compare Logistic Regression, Bidirectional LSTM, and BERT models.
2. Assess the effectiveness of traditional machine learning vs. deep learning and transfer learning techniques.
3. Evaluate the impact of preprocessing techniques on model performance.

### **Key Findings**
- **Bidirectional LSTM** achieved the highest accuracy at **99%**.
- **Logistic Regression** and **BERT** performed comparably with accuracies of **95%**.
- Traditional methods remain competitive, while advanced techniques excel with more nuanced tasks.

---

## **Dataset**
- **Name**: Sentiment140
- **Size**: 1,600,000 annotated tweets
- **Sentiment Labels**:
  - **0**: Negative
  - **2**: Neutral
  - **4**: Positive
- **Download Link**: [Sentiment140 Dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## **Methodology**

### **1. Data Preprocessing**
1. **Text Cleaning**:
   - Lowercasing text
   - Handling emojis and HTML entities
   - Removing mentions, punctuation, and stopwords
2. **Tokenization and Lemmatization**:
   - Tokenize tweets into words
   - Lemmatize tokens using WordNetLemmatizer
3. **Contraction Expansion**:
   - Expand common contractions (e.g., "can't" to "cannot").
4. **Sentiment Mapping**:
   - Convert original labels (0, 2, 4) to (0: Negative, 1: Neutral, 2: Positive).

### **2. Exploratory Data Analysis (EDA)**
- Visualized word frequencies with bar charts and word clouds.
- Analyzed sentiment distribution with pie and bar charts.

### **3. Feature Engineering**
1. **Logistic Regression**:
   - TF-IDF vectorization to convert text into numerical feature vectors.
2. **LSTM**:
   - Tokenization and padding to create sequences of uniform length.
   - Embedding layer for dense vector representations.
3. **BERT**:
   - Pre-trained BERT tokenizer for context-aware embeddings.

### **4. Model Deployment**
1. **Logistic Regression**:
   - Achieved **95% accuracy** with TF-IDF features.
2. **LSTM**:
   - Implemented with embedding and LSTM layers.
   - Accuracy: **99%**.
3. **BERT**:
   - Fine-tuned pre-trained BERT model.
   - Achieved **95% accuracy** with limited fine-tuning.

---

## **Results**
### **Comparison of Models**
| Model              | Accuracy | Strengths                                                                                   | Weaknesses                                                |
|--------------------|----------|--------------------------------------------------------------------------------------------|----------------------------------------------------------|
| Logistic Regression| 95%      | Simple, efficient, and competitive with TF-IDF features.                                   | Struggles with contextual dependencies.                  |
| Bidirectional LSTM | 99%      | Captures forward and backward dependencies effectively.                                    | Computationally intensive.                               |
| BERT               | 95%      | Context-aware embeddings; robust on complex NLP tasks.                                    | High computational requirements and fine-tuning complexity.|

---

## **Insights**
- **LSTM's Performance**:
  - Outperformed other models due to its ability to capture temporal dependencies and context.
- **Challenges with BERT**:
  - Requires significant computational resources for fine-tuning on large datasets.
- **Logistic Regression's Competitiveness**:
  - Simple yet effective baseline for sentiment analysis tasks.

---

## **Future Improvements**
1. **Hyperparameter Tuning**:
   - Optimize learning rates, dropout rates, and LSTM units.
2. **Pretrained Word Embeddings**:
   - Incorporate GloVe or Word2Vec embeddings for enhanced LSTM performance.
3. **Ensemble Models**:
   - Combine Logistic Regression, LSTM, and BERT for improved generalizability.
4. **Dataset Expansion**:
   - Use multilingual or larger datasets for broader analysis.
5. **Model Interpretability**:
   - Use SHAP or LIME to explain predictions for real-world applications.

---

## **Usage**
### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/username/twitter-sentiment-analysis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### **Running the Models**
1. **Logistic Regression**:
   ```bash
   python logistic_regression.py
   ```
2. **LSTM**:
   ```bash
   python lstm_model.py
   ```
3. **BERT**:
   ```bash
   python bert_model.py
   ```

### **Predictions**
Provide sample tweets to predict sentiments using the pre-trained models:
```bash
python predict.py
```

---

## **Technologies Used**
- Python
- TensorFlow/Keras
- PyTorch
- Scikit-learn
- Hugging Face Transformers
- NLTK
- Pandas, NumPy, Matplotlib, Seaborn

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for more details.

---

## **Acknowledgments**
- Sentiment140 Dataset creators
- Hugging Face for the BERT model and tokenizer
- Keras and TensorFlow for LSTM model development

