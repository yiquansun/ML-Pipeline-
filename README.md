Census Income Prediction with FastAPI on Render
Project Overview

This project implements a production-ready machine learning pipeline to predict whether an individual's income exceeds $50K based on 1994 Census data. It features a cleaned data pipeline, a trained Random Forest model, and a RESTful API deployed via CI/CD.
1. Environment Setup

    Python Version: 3.8+ (Conda environment census_env)

    Virtual Environment:
    Bash

    conda activate census_env
    pip install -r requirements.txt

    Key Dependencies: FastAPI, Uvicorn, Scikit-Learn, DVC, Pandas.

2. Data & Version Control

    Data: The dataset is a cleaned version of the UCI Census Income dataset.

    DVC: Data and model artifacts are tracked using DVC.

    Cleaning: The original data was processed to remove leading/trailing spaces and handle missing values.

3. Model Training & Evaluation

    Training: Run python train_model.py to train the model and save artifacts to the /model directory.

    Slicing: The script slice_performance.py calculates performance metrics on categorical data slices to check for model bias.

    Unit Tests: Coverage includes tests for data processing, model inference, and API response integrity using pytest.

4. API Documentation

The API is built with FastAPI and uses Pydantic for data validation, specifically utilizing aliases to handle hyphenated census field names (e.g., education-num).

    Root GET: Returns a welcome message.

    Predict POST: Ingests census data in JSON format and returns a prediction (<=50K or >50K).

    Live Documentation: https://ml-pipeline-jmr6.onrender.com/docs

5. Deployment & CI/CD

    Platform: Render.

    Continuous Integration: GitHub Actions automatically runs pytest and flake8 on every push to the master branch.

    Continuous Deployment: Render automatically redeploys the service only after CI tests pass.

6. Screenshots

Include the following images in your submission folder or embed them here:

    continuous_integration.png: Showing passing GitHub Actions.

    Render_Deployment.png: Showing the "Live" status on the Render dashboard.

    Live_post.png: Showing the live_post.py output with a 200 Status Code.
	
	example.png: REST API