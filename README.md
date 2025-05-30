# MACHINE-LEARNING-MODEL-IMPLEMENTATION

"COMPANY": CODTECH IT SOLUTIONS

"NAME": TEJAS K

"INTERN ID": CT12WN103

"DOMAIN": PYTHON PROGRAMMING

"DURATION": 12 WEEKS

"MENTOR": NEELA SANTHOSH

"DESCRIPTION"

Spam Email Classifier Using Naive Bayes in Python
This script implements a Spam Email Classifier using Machine Learning in Python. The goal is to distinguish between spam and non-spam (ham) messages using a dataset of SMS texts. The entire implementation was executed in Visual Studio Code (VS Code), using a Jupyter Notebook environment, which allows step-by-step development and testing in a modular and interactive manner.

Overview of the Workflow
The script follows a structured machine learning pipeline consisting of data loading, preprocessing, feature extraction, model training, prediction, and evaluation.
  1.	Libraries and Tools Used:
    o	pandas and numpy for data manipulation and numerical computations.
    o	matplotlib.pyplot and seaborn for visualizations.
    o	scikit-learn (sklearn) for machine learning tools like model training, feature extraction, and evaluation metrics.
  2.	Dataset:
    o	The dataset is sourced from a public GitHub repository, originating from the UCI Machine Learning Repository.
    o	It is a tab-separated file containing two columns: the label (ham or spam) and the SMS message.
    o	The script loads the dataset using pandas.read_csv() and assigns column names.
  3.	Data Preprocessing:
    o	The target labels ham and spam are converted into numerical format using a mapping: ham → 0 and spam → 1.
    o	This transformation makes the labels usable by machine learning algorithms.
  4.	Train-Test Split:
    o	The data is split into a training set (70%) and a test set (30%) using train_test_split() from sklearn.model_selection.
    o	This ensures that model evaluation is performed on unseen data, giving a realistic estimate of performance.
  5.	Feature Extraction using TF-IDF:
    o	Text data is converted into numeric features using the TfidfVectorizer from sklearn.feature_extraction.text.
    o	TF-IDF (Term Frequency-Inverse Document Frequency) gives a numerical weight to words based on their importance and frequency in the corpus, helping the model focus on more meaningful words.
  6.	Model Training - Naive Bayes:
    o	A MultinomialNB (Multinomial Naive Bayes) classifier from sklearn.naive_bayes is used.
    o	This algorithm is particularly effective for text classification tasks like spam detection due to its simplicity and efficiency.
  7.	Model Evaluation:
    o	The model’s accuracy is calculated using accuracy_score.
    o	A classification_report is generated, showing precision, recall, and F1-score for both classes.
    o	A confusion_matrix is also displayed to analyze the number of correct and incorrect predictions.
  8.	Visualization:
    o	A heatmap of the confusion matrix is plotted using seaborn.heatmap, providing an intuitive visualization of the classifier’s performance.

Applications
This model has broad real-world applicability in domains such as:
  •	Email Spam Detection: Automatically filtering spam from inboxes.
  •	SMS Filtering: Blocking promotional or scam messages on smartphones.
  •	Customer Support: Classifying support queries into spam or legitimate concerns.
  •	Content Moderation: Identifying unsolicited or harmful messages on messaging platforms.

With further tuning and more data, this model can be adapted to filter different types of textual content, enhancing digital communication safety and user experience.

