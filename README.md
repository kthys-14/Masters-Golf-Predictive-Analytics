# Golf Tournament Winner Predictor

A machine learning application that predicts the most likely winners of PGA Tour tournaments based on player statistics and course characteristics.

## Overview

This application uses historical player performance data and course characteristics to predict which players are most likely to win at specific golf tournaments. The model analyzes several key performance metrics including:

- Strokes gained metrics (approach, around green, off the tee, putting)
- Driving distance and accuracy
- Greens in regulation
- Rough and fairway proximity
- Round scoring

## Installation

1. Ensure you have Python 3.7+ installed
2. Clone this repository
3. Run the setup script to install dependencies and create the necessary directories:
   ```
   python setup.py
   ```

## Usage

1. Launch the Streamlit application:
   ```
   streamlit run golf_predictor.py
   ```

2. In the application:
   - Click "Fetch Data from DataGolf API" to load the data
   - Click "Train Prediction Model" to build the machine learning model
   - Select a golf course to predict tournament winners
   - Optionally, search for specific players

## Data Source

This application uses data from the [DataGolf API](https://datagolf.com/), which provides comprehensive statistics for professional golf. The application can also generate synthetic data for demonstration purposes.

## Model Details

The application trains and evaluates multiple machine learning models:
- Logistic Regression
- Random Forest
- Gradient Boosting

The best performing model (based on AUC score) is automatically selected for making predictions.

## Handling Class Imbalance

Since tournament wins are rare events (class imbalance), the models use techniques like:
- Class weighting
- SMOTE (Synthetic Minority Over-sampling Technique)
- Cost-sensitive evaluation metrics

This ensures the model can effectively identify players with higher win probabilities despite the rareness of wins in the dataset.

## Features

- Real-time data fetching from DataGolf API
- Interactive course selection
- Player search functionality 
- Visual representation of win probabilities
- Model performance metrics
- Course information display 