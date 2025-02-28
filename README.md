# Sentiment Analysis with Naive Bayes

This project uses a **Naive Bayes** classifier to predict the sentiment (positive/negative) of a given text. The model is trained using a **Sentiment Analysis dataset**, which contains labeled text data for sentiment classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [API Endpoint](#api-endpoint)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project leverages **Naive Bayes** for sentiment classification. The model was trained on a publicly available Sentiment Analysis dataset, which includes labeled text data (positive or negative sentiment). 

The trained model and **CountVectorizer** are served via a **Flask API**. This allows you to interact with the model and analyze the sentiment of text inputs in real-time via a web interface.

## Installation

To run this project locally, follow these steps:

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis-naive-bayes.git
cd sentiment-analysis-naive-bayes

