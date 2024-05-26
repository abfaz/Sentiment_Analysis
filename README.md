# Sentiment Analysis of Real-time Flipkart Product Reviews

## Objective

The objective of this project is to classify customer reviews as positive or negative and understand the pain points of customers who write negative reviews. By analyzing the sentiment of reviews, we aim to gain insights into product features that contribute to customer satisfaction or dissatisfaction.

## Dataset

A team of Data Engineers have already scraped real-time data from Flipkart website. The dataset consists of 8,518 reviews for the "YONEX MAVIS 350 Nylon Shuttle" product from Flipkart. Each review includes features such as:

- Reviewer Name
- Rating
- Review Title
- Review Text
- Place of Review
- Date of Review
- Up Votes
- Down Votes

## Data Preprocessing

1. **Text Cleaning**: Remove special characters, punctuation, and stopwords from the review text.
2. **Text Normalization**: Perform lemmatization or stemming to reduce words to their base forms.
3. **Numerical Feature Extraction**: Apply techniques like Bag-of-Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF) for feature extraction.

## Modeling Approach

1. **Model Selection**: Train and evaluate various machine learning models like Naive Bayes, Logistic Regression, and Decision Tree using the embedded text data.
2. **Evaluation Metric**: Use the F1-Score as the evaluation metric to assess the performance of the models in classifying sentiment.

## Model Deployment

1. **Flask App Development**: Develop a Flask web application that takes user input in the form of a review and generates the sentiment (positive or negative) of the review.
2. **Model Integration**: Integrate the trained sentiment classification model into the Flask app for real-time inference.
3. **Deployment**: Deploy the Flask app on an AWS EC2 instance to make it accessible over the internet.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/sentiment-analysis-flipkart-reviews.git
    cd sentiment-analysis-flipkart-reviews
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```sh
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Train the model**: 
    ```sh
    python train_model.py
    ```

2. **Run the Flask application**:
    ```sh
    python app.py
    ```

3. **Open your web browser** and navigate to `http://127.0.0.1:5000/` to use the sentiment analysis application.


## License

This project is licensed under the MIT License.




