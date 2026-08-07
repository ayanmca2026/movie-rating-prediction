# Step-by-Step Deployment Guide

This guide explains how to deploy the **Movie Rating Prediction** Streamlit application to the web so that anyone can access it.

We will focus on **Streamlit Community Cloud**, which is the easiest, fastest, and completely free method to deploy Streamlit apps directly from your GitHub repository.

---

## Method 1: Deploying to Streamlit Community Cloud (Recommended)

Streamlit Community Cloud links directly to your public GitHub repository, detects dependencies, and hosts the app for free.

### Step 1: Set Up Your Accounts
1. Ensure your code is pushed to your public GitHub repository:  
   `https://github.com/ayanmca2026/movie-rating-prediction`
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **Sign Up** or **Sign In**.
3. Choose **Continue with GitHub** to link your Streamlit and GitHub accounts.

### Step 2: Deploy Your App
1. Once signed in, click the **"New app"** button in the top right corner of the dashboard.
2. In the deployment form, fill in the details:
   - **Repository:** `ayanmca2026/movie-rating-prediction`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
3. (Optional) Customize the app URL in the **"Custom URL"** field (e.g., `movie-rating-predict-ayan.streamlit.app`).

### Step 3: Launch
1. Click **Deploy!**
2. Streamlit will launch a server, clone your repository, build the dependencies from `requirements.txt`, and launch your app.
3. You will see a live console log on the right side of the screen showing the build progress.
4. Within 1-2 minutes, your application will be live at the custom Streamlit URL!

---

## Method 2: Deploying to Hugging Face Spaces (Alternative)

Hugging Face Spaces is another free cloud hosting service that natively supports Streamlit applications.

### Step 1: Create a Space
1. Go to [huggingface.co](https://huggingface.co/) and sign in.
2. Click on your profile image in the top-right and select **"New Space"**.
3. Fill in the details:
   - **Space Name:** `movie-rating-prediction`
   - **License:** `mit`
   - **SDK:** Select **Streamlit**.
   - **Space Hardware:** Select **Cpu Basic (Free)**.
   - **Visibility:** Public.
4. Click **Create Space**.

### Step 2: Push Your Code to Hugging Face
Hugging Face Spaces act as remote Git repositories. You can add your Space as a Git remote and push to it:
1. Copy the Git clone URL of your Space.
2. In your terminal inside `movie-rating-prediction/`, link your Hugging Face Space:
   ```bash
   git remote add hf https://huggingface.co/spaces/<your-username>/movie-rating-prediction
   ```
3. Push your files:
   ```bash
   git push hf main
   ```
4. Hugging Face will automatically detect `requirements.txt` and launch your Streamlit app!

---

## Troubleshooting Deployment Errors

### 1. "ModuleNotFoundError: No module named 'src'"
- **Cause:** When deployed, Streamlit starts executing from `app/app.py`. If python path doesn't include the project root, it won't be able to import modules from `src/`.
- **Solution:** We already resolved this in our `app/app.py` code by dynamically appending the parent directory to Python's search path:
  ```python
  import sys
  from pathlib import Path
  sys.path.append(str(Path(__file__).resolve().parent.parent))
  ```

### 2. "Model file not found"
- **Cause:** You did not push `models/movie_rating_model.pkl` to your repository, or the file path is incorrect.
- **Solution:** Make sure you ran `git push` for the `models/` directory. Check that the path `models/movie_rating_model.pkl` is exactly mapped in your codebase.

### 3. App Runs Out of Memory
- **Cause:** Too many large libraries or cache storage issues.
- **Solution:** The `requirements.txt` contains only lightweight standard dependencies (pandas, numpy, scikit-learn, streamlit, matplotlib, seaborn, joblib, pytest). HistGradientBoosting is memory-efficient, so it will run comfortably on free tier CPU plans (which typically offer 16GB of RAM).
