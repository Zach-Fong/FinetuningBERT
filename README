
# Finetuning DistilBERT

## Project Overview

This project explores the application of DistilBERT integrated with various regression heads to predict the popularity of comments within the AskReddit subreddit. By leveraging natural language processing (NLP) and machine learning techniques, the goal is to understand the effectiveness of a distilled transformer model in analyzing textual content for sentiment and popularity. The project tests different regression head architectures—ranging from simple linear layers to more complex Multi-Layer Perceptrons (MLP) and LSTM networks—to explore the most effective approach in predicting comment scores.

## Table of Contents

- Background and Motivation
- Task Definition
- Project Structure
- Data Collection and Preprocessing
- Model Architectures
- Experiments and Results
- Future Work
- References

## Background and Motivation

Social media platforms thrive on user interactions, and engaging comments are crucial for a positive user experience. Predicting comment popularity is influenced by various factors, including content, length, and user engagement. This project focuses on the AskReddit subreddit, where users post questions and receive answers in the form of comments. We investigate whether machine learning models, specifically DistilBERT with regression heads, can effectively predict the relative popularity of these comments.

## Task Definition

The primary task of this project is to predict the relative popularity of Reddit comments within their respective posts. This is achieved by calculating a score ratio, which is the score of each comment divided by the highest scoring comment in the same post. This ratio reflects the relative popularity rather than absolute score values, allowing the model to focus on engagement levels within the context of each post.

The input to the model is a concatenation of the post title and the comment text, tokenized using DistilBERT's uncased tokenizer. The goal is to train a model that can accurately predict this score ratio.

## Project Structure

- **report.pdf**: Contains the detailed report on the project, including the background, methodology, results, and analysis.
- **project.ipynb**: Jupyter Notebook implementing the model training, testing, and evaluation. It includes data loading, preprocessing, model definitions, training loops, and result analysis.
- **get_raw_data.py**: Python script for fetching and preprocessing the raw Reddit data used in this project.

## Data Collection and Preprocessing

The dataset for this project is derived from the AskReddit subreddit, focusing on the top 1100 posts and their associated comments. The data preprocessing involves:
- Tokenizing the combined post title and comment text using DistilBERT's uncased tokenizer.
- Calculating the score ratio for each comment as the ground truth label.
- Splitting the data into training and testing sets, with separate evaluations for within-domain (top 1000 posts) and out-of-domain (posts ranked 1001-1100).

## Model Architectures

Three different regression head configurations were explored:

1. **Single Linear Layer Regression Head**: A baseline model with a simple linear layer mapping DistilBERT's 768-dimensional embeddings to a single output neuron.
2. **Multi-Layer Perceptron (MLP) Regression Head**: A more complex model with four layers, including ReLU activations and dropout for regularization, followed by a final linear layer for regression.
3. **Multi-Layer LSTM**: A custom model replacing DistilBERT, consisting of three LSTM layers with 256 neurons each, followed by a linear output layer.

## Experiments and Results

The models were trained and evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² metrics. Key findings include:
- The MLP regression head outperformed the baseline linear model in capturing complex patterns in the data.
- The LSTM model did not significantly surpass the MLP, but demonstrated potential in processing sequence-based inputs.
- Refining the dataset to balance the score distribution improved the model's ability to predict middle-range and popular comments but introduced challenges in generalization.

### Sample Results

- The linear model exhibited poor performance, often predicting near-zero values across all inputs.
- The MLP model showed better performance but struggled with overfitting, particularly on the refined dataset.
- The LSTM model's performance was comparable to the linear regression head, indicating potential for further exploration.

## Future Work

Future improvements could include:
- Fine-tuning the DistilBERT model along with the regression head to enhance performance.
- Expanding the dataset to include posts from other subreddits or training on a more balanced dataset.
- Exploring BERT models pre-trained on QA-style datasets to improve model accuracy with less fine-tuning.

## References

- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
- Galtier, A. (2021). Fine-tuning BERT for a regression task: is a description enough to predict a property’s list price? ILB Labs publications.
- Yang, R., Cao, J., Wen, Z., Wu, Y., & He, X. (2020). Enhancing Automated Essay Scoring Performance via Fine-tuning Pre-trained Language Models with Combination of Regression and Ranking.
