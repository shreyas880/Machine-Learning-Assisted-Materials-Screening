# Machine-Learning-Assisted-Materials-Screening

## Introduction
This is project is part of a self-learning course that I had taken up, as a part of the Summer Of Science program, hosted by the Maths & Physics Club, IIT Bombay.
I chose to work on Machine Learning applications in the field of Materials Science, as I thought it was quite an intriguing topic.
The key objectives of this project were to learn and apply Machine Learning concepts in a field tying closely to my Major, in Chemistry, and Materials Science is arguably the best possible topic for that objective.

This project demonstrates an end-to-end machine learning pipeline for thermo-mechanical material profiling, screening and ranking. Material property data was collected using the **Material Project API**.


## Project Goals
The main goals for this project were:
- Learning how Machine Learning can be applied to Material Science.
- Understand and get familiar with the end-to-end ML workflow.
- Build a dataset sourced from a scientific database.
- Training, evaluating, and tuning a neural network to predict material properties.
- Develop a simple logical framework to screen and rank materials.

## Repository Structure
**src/**: Contains all the python files.  
**plots/**: Visual Plots to aid in evaluating various processing in the workflow pipeline  
**requirements.txt**

> Note: The full datasets, csv files, as well as pre-trained models are not included in this repository as they can be regenerated using the scripts.

## Project Workflow
The overall project workflow is as follows:
1. Collecting Material Data
2. Elementary Data Analysis
3. Feature Engineering
4. Preprocessing of the Dataset
5. Create Train, Test and Validation sets.
6. Build and train the Neural Network.
7. Evaluate predictions.
8. Create material profiles.
9. Rank candidate materials.

## Dataset
### Data Source:
Material property data was collected from the **Materials Project** database, using the **Materials Project API**.  
The database was chosen because it provides computationally validated properties for a large number of materials, making it well-suited for machine learning applications.

### Data Collection:
Material Properties were retrieved programmatically through the API.
Since the required properties were distributed across multiple API endpoints, data from these endpoints was combined using the material's unique identifier to create a unified dataset.  
The final dataset is automatically created and saved after going through the preprocessing pipeline, rather than being manually assembled.

### Property Selection:
Properties were selected so as to capture both **thermal** and **mechanical** characteristics of materials, aligning with the project's objective of thermo-mechanical screening.  
The selection prioritized properties that were relevant to profiling and ranking candidate materials while maintaining sufficient diversity so as to not induce any external bias.

## Model Development
### Preprocessing:
- The entire dataset was cleaned to remove incomplete/missing entries.
- Relevant features were extracted, numerical preprocessing and feature encoding was performed, if necessary, and the dataset was prepared for neural network training.
- An exploratory data analysis was performed which led to the decision of removing outliers as they had feature values which were not feasible physically.

### Train/Test split:
The standard **80-10-10** split was used to obtain the training, validation and test sets.

### Feature Scaling:
As is common, most of the features did not belong to the same exponential scale, and thus had to be normalized. The _StandardScaler_ from _sklearn.preprocessing_ was used to scale the datasets.
The _fit_transform_ method was used on the training set, while the validation and test sets required the _transform_ method.

### Model Architecture:
- Task: Mutli-target regression to predict 6 target properties simultaneously.
- Architecture: Fully connected, dense, feedforward neural network.
- Number of Hidden Layers: 4
- Layer Sizes: 512 -> 256 -> 128 -> 64 -> 6
- Activation functions used: ReLU.
- Regularization methods: Dropout layers, Early Stopping.
- Output Layer: 6 neurons representing the output for each of the six target properties.

### Evaluation metrics:
#### Training and Validation Loss Curves: <img width="2371" height="1468" alt="loss_curve" src="https://github.com/user-attachments/assets/f9cfd553-edb3-4924-bf45-7b91d21e7a09" />
#### R<sup>2</sup> Scores: <img width="2365" height="1465" alt="R2_scores" src="https://github.com/user-attachments/assets/408dbca6-ee22-42ef-bcc8-fda36d5b8b57" />
#### MAE Scores: <img width="2352" height="1465" alt="MAE_scores" src="https://github.com/user-attachments/assets/34d8b514-7ed6-4604-b153-203980db0e7d" />
#### RMSE Scores: <img width="2352" height="1465" alt="RMSE_scores" src="https://github.com/user-attachments/assets/62e5a976-c95f-4bd9-b73e-3064e27a5657" />
#### Predicted vs Actual Values Graph: <img width="4170" height="2368" alt="predicted_vs_actual" src="https://github.com/user-attachments/assets/968497d6-c69b-4221-8896-bcf8f6c4fae2" />

## Material Ranking Framework  
To build a material profile, was just the beginning, but the incentive of ranking the profiles, still remained. I wanted to keep the framework purely logical and based on how accurately the model could predict each of the target values.  
The profiles are also ranked across multiple categories, i.e., **thermal ranking**, **mechanical ranking**, **overall/balanced ranking**  
Keeping these constrains in mind, this is the logic behind the ranking framework:
- **Predictions to Percentiles**: Each of the predicted properties for a material is turned into a score based on its percentile within the test set.
- **Scoring Strategy**: The final score for each of the three categories, by which I am ranking the materials, is calculated by taking the weighted sum of the property scores and their weights.
> The weights are more or less equally distributed, but some changes were made to give more importance to the properties that were predicted more accurately.
- **Limitations**: The weighted ranking framework is not experimentally validated. 
