import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import streamlit as st
import requests
import json
import os
from PIL import Image
import io
import base64
import time
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

class GolfTournamentPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'round_score', 'sg_app', 'sg_arg', 'sg_ott', 'sg_putt',
            'driving_distance', 'driving_accuracy', 'greens_in_regulation',
            'rough_proximity', 'fairway_proximity'
        ]
        self.course_data = None
        self.player_data = None
        self.course_images = self.load_course_images()
        
    def load_course_images(self):
        """
        Load course images from a directory or create placeholders
        In a real implementation, this would load actual course images
        """
        # Placeholder for course images - in a real implementation, these would be actual images
        courses = [
            "Augusta National Golf Club",
            "TPC Sawgrass",
            "Pebble Beach Golf Links",
            "St Andrews Links",
            "Torrey Pines Golf Course",
            "Bethpage Black Course",
            "Pinehurst Resort",
            "Whistling Straits",
            "Kiawah Island Golf Resort",
            "Winged Foot Golf Club"
        ]
        
        # Create a dictionary mapping course names to placeholder images
        course_images = {}
        for course in courses:
            # In a real implementation, this would load from files
            # For simulation, we'll create colored placeholders
            course_images[course] = f"Image placeholder for {course}"
        
        return course_images
    
    def load_datagolf_api_data(self, api_key=None):
        """
        Fetch data from the DataGolf API
        This is a placeholder - in a real implementation, this would make actual API calls
        """
        print("Fetching data from DataGolf API...")
        
        # Simulate API data fetch - in real implementation, use requests to fetch from DataGolf API
        # Example: response = requests.get(f"https://api.datagolf.com/historical-data?key={api_key}")
        
        # Generate synthetic data for development purposes
        self.generate_synthetic_data()
        
        print("Data fetched successfully!")
        
    def generate_synthetic_data(self):
        """
        Generate synthetic data for development purposes
        In a real implementation, this would be replaced with actual API calls
        """
        # Generate player data
        players = [
            "Tiger Woods", "Rory McIlroy", "Brooks Koepka", "Dustin Johnson", 
            "Justin Thomas", "Jon Rahm", "Collin Morikawa", "Bryson DeChambeau",
            "Jordan Spieth", "Xander Schauffele", "Patrick Cantlay", "Viktor Hovland",
            "Scottie Scheffler", "Will Zalatoris", "Max Homa", "Tony Finau",
            "Matt Fitzpatrick", "Cameron Smith", "Hideki Matsuyama", "Shane Lowry"
        ]
        
        courses = list(self.course_images.keys())
        
        # Generate random player stats
        player_data = []
        for player in players:
            for _ in range(20):  # 20 tournament entries per player
                # Add some consistent player style by adding a bias to certain players
                if player in ["Tiger Woods", "Rory McIlroy", "Dustin Johnson"]:
                    distance_bias = np.random.normal(15, 5)  # These players hit longer
                    accuracy_bias = np.random.normal(-5, 2)  # But less accurate
                elif player in ["Jordan Spieth", "Matt Fitzpatrick", "Shane Lowry"]:
                    distance_bias = np.random.normal(-10, 5)  # These players hit shorter
                    accuracy_bias = np.random.normal(10, 2)  # But more accurate
                else:
                    distance_bias = np.random.normal(0, 3)
                    accuracy_bias = np.random.normal(0, 3)
                
                # Randomize which course
                course = np.random.choice(courses)
                
                # Generate random stats with some correlation and player bias
                round_score = np.random.normal(72, 3)
                sg_app = np.random.normal(0.2, 0.8)
                sg_arg = np.random.normal(0.1, 0.7)
                sg_ott = np.random.normal(0.3, 0.9) + distance_bias/20
                sg_putt = np.random.normal(0.1, 1.2)
                driving_distance = np.random.normal(295, 15) + distance_bias
                driving_accuracy = np.random.normal(65, 8) + accuracy_bias
                greens_in_regulation = np.random.normal(70, 10) + accuracy_bias/2
                rough_proximity = np.random.normal(35, 5)
                fairway_proximity = np.random.normal(25, 3)
                
                # Determine win based on a combination of factors
                # Simulate course fit: certain players perform better on certain courses
                course_fit = np.random.normal(0, 1)
                if player in ["Tiger Woods", "Jordan Spieth"] and "Augusta" in course:
                    course_fit += 2.0  # Tiger and Jordan play well at Augusta
                elif player in ["Rory McIlroy", "Brooks Koepka"] and "Bethpage" in course:
                    course_fit += 1.5  # Rory and Brooks play well at Bethpage
                
                # Calculate win probability based on stats and course fit
                win_probability = (
                    -0.2 * round_score + 
                    0.6 * sg_app + 
                    0.4 * sg_arg + 
                    0.5 * sg_ott + 
                    0.4 * sg_putt + 
                    0.002 * driving_distance + 
                    0.01 * driving_accuracy + 
                    0.01 * greens_in_regulation + 
                    -0.01 * rough_proximity + 
                    -0.02 * fairway_proximity +
                    0.5 * course_fit
                )
                
                # Normalize to probability scale and introduce randomness
                win_probability = 1 / (1 + np.exp(-win_probability))
                win = 1 if np.random.random() < win_probability * 0.05 else 0  # Make wins rare
                
                player_data.append({
                    'player_name': player,
                    'course': course,
                    'round_score': round_score,
                    'sg_app': sg_app,
                    'sg_arg': sg_arg,
                    'sg_ott': sg_ott,
                    'sg_putt': sg_putt,
                    'driving_distance': driving_distance,
                    'driving_accuracy': driving_accuracy,
                    'greens_in_regulation': greens_in_regulation,
                    'rough_proximity': rough_proximity,
                    'fairway_proximity': fairway_proximity,
                    'win': win
                })
        
        self.player_data = pd.DataFrame(player_data)
        
        # Generate course characteristics data
        course_data = []
        for course in courses:
            if "Augusta" in course:
                style_bias = "Distance"
            elif "TPC Sawgrass" in course:
                style_bias = "Accuracy"
            elif "Pebble Beach" in course:
                style_bias = "Short Game"
            elif "St Andrews" in course:
                style_bias = "Links"
            else:
                style_bias = np.random.choice(["Distance", "Accuracy", "Short Game", "Balanced"])
                
            course_data.append({
                'course_name': course,
                'course_length': np.random.randint(7000, 7800),
                'fairway_width': np.random.normal(30, 5),
                'rough_length': np.random.normal(3, 1),
                'green_speed': np.random.normal(12, 1),
                'green_size': np.random.normal(5000, 1000),
                'bunker_count': np.random.randint(40, 100),
                'water_hazards': np.random.randint(0, 15),
                'elevation_changes': np.random.normal(50, 20),
                'style_bias': style_bias
            })
        
        self.course_data = pd.DataFrame(course_data)
        
    def preprocess_data(self):
        """
        Preprocess the data for model training
        """
        if self.player_data is None:
            raise ValueError("Data not loaded. Call load_datagolf_api_data() first.")
        
        # Select features and target
        X = self.player_data[self.feature_names]
        y = self.player_data['win']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """
        Train a logistic regression model with class weight adjustment
        """
        # Handle class imbalance with class weights
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def train_random_forest(self, X_train, y_train):
        """
        Train a random forest model with class weight adjustment
        """
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight='balanced',
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """
        Train a gradient boosting model
        """
        # For GBM, we'll use SMOTE to handle class imbalance
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42
        )
        model.fit(X_train_resampled, y_train_resampled)
        return model
    
    def train_and_evaluate_models(self):
        """
        Train multiple models and evaluate their performance
        """
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        # Train models
        models = {
            'Logistic Regression': self.train_logistic_regression(X_train, y_train),
            'Random Forest': self.train_random_forest(X_train, y_train),
            'Gradient Boosting': self.train_gradient_boosting(X_train, y_train)
        }
        
        # Evaluate models
        results = {}
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }
            
            print(f"Model: {name}")
            print(f"Accuracy: {results[name]['accuracy']:.4f}")
            print(f"Precision: {results[name]['precision']:.4f}")
            print(f"Recall: {results[name]['recall']:.4f}")
            print(f"F1 Score: {results[name]['f1']:.4f}")
            print(f"AUC: {results[name]['auc']:.4f}")
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            print("\n")
        
        # Select the best model based on AUC (good for imbalanced datasets)
        best_model_name = max(results, key=lambda x: results[x]['auc'])
        self.model = models[best_model_name]
        print(f"Best model: {best_model_name} with AUC = {results[best_model_name]['auc']:.4f}")
        
        # Feature importance (for tree-based models)
        if best_model_name in ['Random Forest', 'Gradient Boosting']:
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print("Feature importance:")
            for i in indices:
                print(f"{self.feature_names[i]}: {importances[i]:.4f}")
        
        return results
    
    def predict_win_probability(self, player_stats):
        """
        Predict win probability for a player based on their stats
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_and_evaluate_models() first.")
        
        # Scale the input
        stats_array = np.array([player_stats[feature] for feature in self.feature_names]).reshape(1, -1)
        stats_scaled = self.scaler.transform(stats_array)
        
        # Predict probability
        win_prob = self.model.predict_proba(stats_scaled)[0, 1]
        return win_prob
    
    def predict_tournament_winners(self, course_name):
        """
        Predict win probabilities for all players at a specific course
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_and_evaluate_models() first.")
        
        # Get unique players
        players = self.player_data['player_name'].unique()
        
        # For each player, get their average stats and predict win probability
        predictions = []
        for player in players:
            # Get player's average stats
            player_stats = self.player_data[self.player_data['player_name'] == player][self.feature_names].mean().to_dict()
            
            # Predict win probability
            win_prob = self.predict_win_probability(player_stats)
            
            # Add some course-specific adjustment
            # This is a simplification - in a real model, this would be more sophisticated
            if course_name in self.player_data[self.player_data['player_name'] == player]['course'].values:
                # Player has experience at this course
                player_at_course = self.player_data[(self.player_data['player_name'] == player) & 
                                                     (self.player_data['course'] == course_name)]
                if not player_at_course.empty:
                    course_performance = player_at_course['win'].mean() * 2  # Boost if they've won here before
                    win_prob = win_prob * (1 + course_performance)
            
            predictions.append({
                'player': player,
                'win_probability': win_prob
            })
        
        # Sort by win probability
        predictions = sorted(predictions, key=lambda x: x['win_probability'], reverse=True)
        return predictions

# Streamlit App
def run_streamlit_app():
    st.set_page_config(page_title="Golf Tournament Winner Predictor", page_icon="🏌️", layout="wide")
    
    st.title("🏌️ PGA Tour Tournament Winner Predictor")
    st.markdown("""
    This application predicts the most likely winners of a PGA Tour tournament based on player statistics and course characteristics.
    """)
    
    # Initialize the predictor
    predictor = GolfTournamentPredictor()
    
    # Sidebar for data loading and model training
    with st.sidebar:
        st.header("Model Controls")
        
        if st.button("Load Data"):
            with st.spinner("Loading data from DataGolf API..."):
                predictor.load_datagolf_api_data()
            st.success("Data loaded successfully!")
        
        if predictor.player_data is not None:
            if st.button("Train Model"):
                with st.spinner("Training and evaluating models..."):
                    results = predictor.train_and_evaluate_models()
                
                st.success("Model trained successfully!")
                
                # Display model metrics
                st.header("Model Performance")
                metrics_df = pd.DataFrame(results).T
                st.dataframe(metrics_df.style.highlight_max(axis=0))
    
    # Main area for predictions
    if predictor.model is not None:
        st.header("Tournament Winner Predictions")
        
        # Course selection
        courses = list(predictor.course_images.keys())
        selected_course = st.selectbox("Select a golf course:", courses)
        
        # Player search box
        search_player = st.text_input("Search for a specific player:")
        
        # Make predictions
        with st.spinner("Calculating win probabilities..."):
            predictions = predictor.predict_tournament_winners(selected_course)
        
        # Display course image (placeholder)
        st.subheader(f"Selected Course: {selected_course}")
        
        # In a real implementation, this would display an actual image
        st.info(f"Course Image: {predictor.course_images[selected_course]}")
        
        # Display predictions
        st.subheader("Predicted Win Probabilities")
        
        # Filter by player search if provided
        if search_player:
            filtered_predictions = [p for p in predictions if search_player.lower() in p['player'].lower()]
            if not filtered_predictions:
                st.warning(f"No players found matching '{search_player}'")
            else:
                predictions = filtered_predictions
        
        # Create DataFrame for display
        predictions_df = pd.DataFrame(predictions)
        predictions_df['win_probability'] = predictions_df['win_probability'] * 100  # Convert to percentage
        predictions_df.columns = ['Player', 'Win Probability (%)']
        
        # Display as table
        st.dataframe(predictions_df.style.format({'Win Probability (%)': '{:.2f}%'}), height=600)
        
        # Visualize top 10 players
        st.subheader("Top 10 Players' Win Probabilities")
        top_10 = predictions_df.head(10)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(top_10['Player'], top_10['Win Probability (%)'], color='skyblue')
        ax.set_xlabel('Win Probability (%)')
        ax.set_ylabel('Player')
        ax.set_title(f'Top 10 Players Most Likely to Win at {selected_course}')
        
        # Add percentage labels on the bars
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.2f}%',
                   ha='left', va='center')
            
        st.pyplot(fig)
    else:
        st.info("Please load data and train the model using the controls in the sidebar.")

if __name__ == "__main__":
    run_streamlit_app()