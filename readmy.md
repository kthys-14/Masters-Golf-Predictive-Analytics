Introduction 

In professional golf, predictive analytics has largely focused on over/under stroke projections, leaving a gap in forecasting which player is most likely to win a specific tournament based on optimal course play and player style. Major tournaments like the PGA Championship feature diverse course layouts that suit different playing styles. Most current predictive models estimate win probabilities only based on stroke information. Our goal is to build a predictive model that can estimate win probabilities before the tournament begins by incorporating playing style and course characteristics, differentiating it from existing models. 

Problem Description 

Current models for predicting professional golf outcomes primarily rely on player statistics, without factoring in how course characteristics influence those outcomes. Our project aims to develop a predictive model that combines player performance data with detailed course features. This integration will allow for more accurate forecasting and offer valuable insights to coaching, broadcasting, and betting strategies. 

Survey of Current Solution 

DataGolf provides a live model that predicts a player's probability of finishing in the top 20, 10, 5, or winning. However, this model updates dynamically during play. 

In “Assessing Golf Performance on the PGA Tour,” Broadie defines course difficulty using detailed shot-level analysis and categorizes difficulty into long-game, short-game, and putting. This contrasts with the USGA's traditional approach for handicapping. 
In “Enhancing PGA Tour Performance: Leveraging ShotLink™ Data for Optimization and Prediction” by Guillot and Stauffer, the authors apply a shortest path model to refine player strategy and predict performance, though it relies solely on ShotLink data and excludes key factors like roll distance. 

In “Determinants of Performance on the PGA Tour,” Peters uses driving accuracy, driving distance, and other performance variables to predict earnings. He also introduces a variable for experience (years on tour), which shows a positive correlation with earnings, favoring veteran players. 

Our model will assess both the optimal way to play a course—defined by top-20 player performance over the past five years—and player performance trends over the same period. The application will display a list of effective courses and corresponding win probabilities for each player. 

Data Source and Dataset Description 
We use DataGolf as our primary data source due to its rich real-time and historical golf data. Our subscription (Scratch PLUS – $270/month) gives us access to detailed PGA Tour event data. 

Key data elements include: 

Golf Ranking: Player rankings. 

Round Scoring, Stats & Strokes Gained: Historical performance data per player, per round, per event. 

Pre-tournament Predictions: DataGolf’s proprietary pre-event performance probabilities. 

Player Details: Country, course, cross-platform IDs (DraftKings, FanDuel), scheduling info (e.g., tee times, rounds). 

Data Acquisition and Preprocessing 
Data is acquired via the DataGolf API. We've successfully retrieved and tested field-level data (e.g., for the Valspar Championship). Preprocessing involves filtering event-specific data, formatting player metrics, and creating dynamic dashboards. No imputation is necessary due to the dataset’s completeness. 

Innovations 

Analyze 10 game-related data points to determine how each influences course strategy. 

Assess player performance across those data points throughout PGA Tour history. 

Recognize that each course presents a unique bias—only certain data points significantly impact performance 

Explore the combination of multiple data points with the players ranking as a predictor 

Modeling Approach 
Our modeling strategy leverages Data Golf’s raw round scoring data and raw historical event data. It contains a total of 10 predictive variables:  

Round score 

Stroke gained: approach the green – sg_app 

Stroke gained: around the green – sg_arg 

Stroke gained: of the tee – sg_ott 

Stroke gained: putting – sg_putt 

Driving distance 

Driving accuracy  

Greens in regulation 

Rough proximity 

Fairway proximity 

Because wins are much less frequent than non-wins in our dataset, we use techniques to make sure our model does not overlook these rare outcomes. For example, we adjust the decision threshold during training so that even a small prediction probability for a win is given extra consideration, and we use cost-sensitive learning—assigning a higher penalty to misclassifying a win (false negatives) than a non-win. This approach forces the model to focus on correctly identifying the win class despite its scarcity. 

We then test several machine learning algorithms: 

Linear regression: Helps determine the impact of independent variables on dependent variables (probability of win). It allows us to find the impact that each of the variables have on the outcome; and while it is easy to train, we could run into an issue if the relationships between the variables are nonlinear. 

Logistic Regression: Offers transparency and ease of interpretation, which is useful for understanding the influence of each variable. 

Random Forests: Can capture non-linear relationships and interactions between variables, often handling imbalanced data more robustly. 

Gradient Boosting: Typically provides high predictive accuracy, especially for datasets where one class is underrepresented. 

To evaluate our models, we use common metrics such as overall accuracy, precision, recall, and f1-score. However, because our primary interest is correctly predicting probability of win, we also incorporate cost-sensitive metrics (like weighted f1-score or custom cost functions) to specifically measure performance on the win class. 

Finally, we validate the model by comparing its predictions against actual outcomes from past tournaments. This real-world testing helps us calibrate the model—adjusting thresholds and cost parameters—to ensure it can be effectively applied in practical settings like coaching, broadcasting, and betting strategies. 

Next Steps and Experiments 
Moving forward, we plan to carry out focused experiments and validation to maximize the effectiveness of our tool: 

Predictive Performance Evaluation: Comparing predicted player rankings and scoring probabilities with real-time tournament outcomes, validating predictive accuracy event-by-event. 

To be able to produce this experiment, we will run the model for three tournaments that have already been played in 2025; WM Phoenix Open, Arnold Palmer Invitational and the Players Championship. We will then compare the accuracy of our model to the actual results from the tournament.  

This will give us a really good sense of where our model sits in comparison to the actual values.   

 

Strategy Optimization: Testing various player-selection strategies based on Data Golf predictions to identify optimal variable influence in final probability.  

Is selecting the top 20 finishers of each tournament the right data set? Or should we select only the top 10? We will evaluate and experiment with our model to determine what is the optimal number of data points that we should evaluate. Additionally, we will run experiments by removing a variable or many variables from the model using their relevance to determine which ones are the most relevant points and the ones that have an actual impact on the result  

Computation and Visualization Components 

Our user interface a simple UI in which the user will be able to select a golf course in the PGA championship, the system will provide the user with a picture of the emblematic course and a list of the players with their win probability sort in descending order. The user will also be able to search for a specific player once a course selection has occurred.   

 

For the UI we are using a Streamlit application making it easy to visualize win probabilities for different players based on course selection.  

 

 

Revised Plan 

We had originally had a 4th team member who would have been contributing towards the data collection and documentation/reporting aspect of the project. Due to unforeseen circumstances, our team member had to drop the course. As such we have taken up some extra responsibilities in their absence and evenly distributed the workload across the remaining team members. Remaining tasks include finishing the model development and data analysis. 

Plans and Activities 

The following list is a breakdown of the activities we will perform to successfully finish our project.  

Data collection and processing 

Saliou 

10h 

Data architecture 

Kattelie  

10h 

Model development and data analysis 

Saliou 

15h 

Visualization and UI development 

Kattelie 

 

10h 

Project management, evaluation design, documentation and reporting 

Akshay 

20h 

Final report and presentation  

All 

25h 

All team members have contributed a similar amount of effort 

 

 

 

References 

Broadie, M. (2012). Assessing Golfer Performance on the PGA TOUR. Interfaces, 42(2), 146–165. 
 
Guillot, M., & Stauffer, G. (2023). Enhancing PGA Tour Performance: Leveraging ShotLink™ Data for Optimization and Prediction. arXiv preprint arXiv:2309.00485. 

Peters, Andrew. Determinants of Performance on the PGA Tour, blogs.elon.edu/ipe/files/2021/02/v17-Andrew-Peters-final.pdf. Accessed 27 Feb. 2025. 

DataGolf. “Raw Data Notes.” DataGolf. Accessed March 23, 2025. https://datagolf.com/raw-data-notes. 

Hoegh, Andrew. (2011). Defining the Performance Coefficient in Golf: A Case Study at the 2009 Masters. Journal of Quantitative Analysis in Sports. 7. 12-12. 10.2202/1559-0410.1331. 

Connolly, Robert & Rendleman, Richard. (2009). Dominance, Intimidation, and 'Choking' on the PGA Tour. Journal of Quantitative Analysis in Sports. 5. 6-6. 10.2202/1559-0410.1161. 

NIX, CHARLES. (1991). Physical Skill Factors Contributing to Success on the Professional Golf Tour. Perceptual and Motor Skills - PERCEPT MOT SKILLS. 72. 10.2466/PMS.72.4.1272-1274. 

Smith, A., Brown, J., & Davis, R. (2019). Neural networks in sports performance prediction. Journal of Sports Analytics, 5(2), 123–134. 

Johnson, M., Thompson, S., & Carter, D. (2018). Comparative analysis of golf predictive models. International Journal of Golf Science, 7(1), 45–60. 
