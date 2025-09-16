import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_regression, make_classification

# Set page config
st.set_page_config(
    page_title="ML Regression Demo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to make metrics smaller
st.markdown("""
<style>
[data-testid="metric-container"] {
    background-color: rgba(28, 131, 225, 0.1);
    border: 1px solid rgba(28, 131, 225, 0.1);
    padding: 5% 5% 5% 10%;
    border-radius: 5px;
    color: rgb(30, 103, 119);
    overflow-wrap: break-word;
}

[data-testid="metric-container"] > div {
    width: fit-content;
    margin: auto;
}

[data-testid="metric-container"] label {
    width: fit-content;
    margin: auto;
    font-size: 0.8rem !important;
}

[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.title("Demo 0: Basic Regression")
st.markdown("---")

# Introduction section
st.markdown("""
This interactive demo demonstrates two fundamental regression techniques with detailed explanations:

1. **Linear Regression** - for continuous target variables (predicting numbers)
2. **Logistic Regression** - for binary classification (yes/no predictions)

""")

demo_type = "Both"

# Linear Regression Section
if demo_type in ["Linear Regression", "Both"]:
    st.markdown("---")
    st.header("Part 1: Linear Regression Demo")
    
    st.markdown("""
    **What is Linear Regression?**  
    Linear regression is like drawing the best straight line through a bunch of points on a graph. 
    It helps us predict a number (like house price) based on other numbers (like house size).
    
    **Real-world examples:**
    - Predicting house prices based on size
    - Predicting temperature based on time of day  
    - Predicting sales based on advertising spend
    """)
    
    # Parameters for linear regression
    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples_linear = st.slider("Number of data points", 50, 200, 100, key="linear_samples")
    with col2:
        noise_level = st.slider("Noise level", 1, 20, 10, key="linear_noise")
    with col3:
        slope_a = st.slider("Slope (a) in y = a*x + b", -30, 30, 10, key="slope_coefficient")
    
    # Initialize session state for random seed if not exists
    if 'random_seed' not in st.session_state:
        st.session_state.random_seed = 42
    
    # Add random sampling button
    st.subheader("1.1 Data Preparation")
    
    if st.button("Random Sampling", type="secondary", help="Resample the train/test split with different random seed"):
        # Generate new random seed and store in session state
        st.session_state.random_seed = np.random.randint(0, 10000)
        st.success(f"Data resampled with new random seed: {st.session_state.random_seed}")
        # Force rerun to update the visualization
        st.rerun()
    
    # Generate linear regression data with custom slope
    # Use a fixed seed for data generation to keep base data consistent
    #np.random.seed(42)  
    X_linear = np.random.normal(0, 1, (n_samples_linear, 1))  # Generate random X values
    # Create y values using our custom slope: y = slope_a * x + intercept + noise
    intercept_b = 5  # Fixed intercept value
    noise = np.random.normal(0, noise_level, n_samples_linear)  # Generate noise
    y_linear = slope_a * X_linear.flatten() + intercept_b + noise
    
    # Convert to DataFrame
    df_linear = pd.DataFrame({
        'feature': X_linear.flatten(),
        'target': y_linear
    })
    
    # Split data using current random seed (this will change when button is pressed)
    test_size = 0.2  # Fixed 20% test size
    X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split(
        X_linear, y_linear, test_size=test_size, random_state=st.session_state.random_seed
    )
    
    # Visualize split data
    st.subheader("1.2 Data Visualization - Train/Test Split")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_train_linear, y_train_linear, alpha=0.6, color='red', label=f'Training Data ({len(X_train_linear)} points)', s=50)
    ax.scatter(X_test_linear, y_test_linear, alpha=0.6, color='green', label=f'Test Data ({len(X_test_linear)} points)', s=50)
    ax.set_xlabel('Feature Value (e.g., House Size)')
    ax.set_ylabel('Target Value (e.g., House Price)')
    ax.set_title(f'Linear Regression Dataset - Train/Test Split (Seed: {st.session_state.random_seed})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate y-limits based on actual data range with some padding

    st.info(f"""
    **What you're seeing:**
    - Red circles represent training data points ({len(X_train_linear)} points used to teach the model)
    - Green circles represent test data points ({len(X_test_linear)} points used to evaluate the model)
    - The X-axis shows our input feature (like house size)
    - The Y-axis shows our target value (like house price)
    - Current random seed: {st.session_state.random_seed}
    """)

    y_min, y_max = y_linear.min(), y_linear.max()
    y_padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    plt.ylim(-50, 50)
    plt.xlim(-5, 5)
    st.pyplot(fig)
        
    # Train model
    linear_model = LinearRegression()
    linear_model.fit(X_train_linear, y_train_linear)
    y_pred_linear = linear_model.predict(X_test_linear)
    
    # Calculate metrics
    mse = mean_squared_error(y_test_linear, y_pred_linear)
    r2 = r2_score(y_test_linear, y_pred_linear)

    # Visualize results
    st.subheader("1.3 Model Visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training data with regression line
    ax1.scatter(X_train_linear, y_train_linear, alpha=0.6, color='red', label='Training Data')
    
    # Create smooth line for predictions
    X_line = np.linspace(X_linear.min(), X_linear.max(), 100).reshape(-1, 1)
    y_line = linear_model.predict(X_line)
    ax1.plot(X_line, y_line, color='blue', linewidth=3, label='Regression Line')
    
    ax1.set_xlabel('Feature Value')
    ax1.set_ylabel('Target Value')
    ax1.set_title('Training Data with Learned Line')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-50, 50)
    ax1.set_xlim(-5, 5)

    # Plot 2: Actual vs Predicted
    ax2.scatter(y_test_linear, y_pred_linear, alpha=0.7, color='green', s=60)
    
    # Add vertical lines to show prediction errors
    for i in range(len(y_test_linear)):
        ax2.plot([y_test_linear[i], y_test_linear[i]], 
                [y_test_linear[i], y_pred_linear[i]], 
                'r-', alpha=0.5, linewidth=1)
    
    #ax2.scatter(X_train_linear, y_train_linear, alpha=0.6, color='blue', label='Training Data')
    ax2.plot([y_test_linear.min(), y_test_linear.max()], 
             [y_test_linear.min(), y_test_linear.max()], 
             'b--', lw=2, label='Perfect Predictions')
    
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title(f'Actual vs Predicted (R² = {r2:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-30, 30)
    ax2.set_xlim(-30, 30)

    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("1.4 Model Training Results")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Squared Error", f"{mse:.2f}", help="Average prediction error (lower is better)")
    with col2:
        st.metric("R² Score", f"{r2:.3f}", help="How well our line fits (1.0 = perfect, 0.0 = no fit)")
    with col3:
        st.metric("Coefficient (Slope)", f"{linear_model.coef_[0]:.3f}", help="How steep our line is")
    with col4:
        st.metric("Intercept", f"{linear_model.intercept_:.3f}", help="Where our line crosses the Y-axis")
    
    # Display learned equation
    st.success(f"**Learned equation:** y = {linear_model.coef_[0]:.3f} × x + {linear_model.intercept_:.3f}")
    st.info(f"This means: for every 1 unit increase in feature, target increases by {linear_model.coef_[0]:.3f}")
        
    st.info(f"""
    **What you're seeing:**
    - **Left plot:** The red line is our model's learned pattern through the blue training data
    - **Right plot:** How close our predictions (Y-axis) are to actual values (X-axis)
    - Points close to the red diagonal line = good predictions
    - Points far from the line = poor predictions
    - Our R² score of {r2:.3f} means our model explains {r2*100:.1f}% of the variation in the data
    """)

# Logistic Regression Section
if demo_type in ["Logistic Regression", "Both"]:
    st.markdown("---")
    st.header("Part 2: Logistic Regression Demo")
    
    st.markdown("""
    **What is Logistic Regression?**  
    Logistic regression is used for binary classification - predicting whether something belongs to one category or another (yes/no, pass/fail, spam/not spam). 
    Instead of drawing a straight line, it creates an S-shaped curve that separates the two classes.
    
    **Real-world examples:**
    - Email spam detection (spam vs not spam)
    - Medical diagnosis (disease vs healthy)  
    - Marketing response (will buy vs won't buy)
    """)
    
    # Parameters for logistic regression
    col1, col2 = st.columns(2)
    with col1:
        n_samples_logistic = st.slider("Number of data points", 50, 200, 100, key="logistic_samples")
    with col2:
        class_sep = st.slider("Class separation", 0.2, 1.0, 0.5, key="class_separation")
    
    # Fixed to 1 feature for simplicity
    n_features = 1
    
    # Initialize session state for logistic regression random seed
    if 'logistic_random_seed' not in st.session_state:
        st.session_state.logistic_random_seed = 42
    
    # Add random sampling button for logistic regression
    st.subheader("2.1 Data Preparation")
    
    if st.button("Random Sampling", type="secondary", help="Resample the train/test split with different random seed", key="logistic_sampling"):
        # Generate new random seed and store in session state
        st.session_state.logistic_random_seed = np.random.randint(0, 10000)
        st.success(f"Data resampled with new random seed: {st.session_state.logistic_random_seed}")
        st.rerun()
    
    # Generate logistic regression data
    X_logistic, y_logistic = make_classification(
        n_samples=n_samples_logistic,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=42
    )
    
    # Convert to DataFrame for display
    df_logistic = pd.DataFrame({
        'feature_1': X_logistic[:, 0],
        'target': y_logistic
    })
    
    # Split data using current random seed
    test_size = 0.2  # Fixed 20% test size
    X_train_logistic, X_test_logistic, y_train_logistic, y_test_logistic = train_test_split(
        X_logistic, y_logistic, test_size=test_size, random_state=st.session_state.logistic_random_seed
    )
    
    # Visualize split data
    st.subheader("2.2 Data Visualization - Train/Test Split")

    st.info(f"""
    **What you're seeing:**
    - Red points represent training data (used to teach the model)
    - Green points represent test data (used to evaluate the model)
    - Circles (○) represent Class 0, Squares (□) represent Class 1
    - The goal is to find a boundary that separates the two classes
    - Current random seed: {st.session_state.logistic_random_seed}
    """)
    
    # 1D visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot training data
    train_class_0 = X_train_logistic[y_train_logistic == 0]
    train_class_1 = X_train_logistic[y_train_logistic == 1]
    ax.scatter(train_class_0, np.zeros_like(train_class_0), alpha=0.6, color='red', 
              label=f'Training Class 0 ({len(train_class_0)} points)', s=50, marker='o')
    ax.scatter(train_class_1, np.ones(len(train_class_1)), alpha=0.6, color='red', 
              label=f'Training Class 1 ({len(train_class_1)} points)', s=50, marker='s')
    
    # Plot test data
    test_class_0 = X_test_logistic[y_test_logistic == 0]
    test_class_1 = X_test_logistic[y_test_logistic == 1]
    ax.scatter(test_class_0, np.zeros_like(test_class_0), alpha=0.6, color='green', 
              label=f'Test Class 0 ({len(test_class_0)} points)', s=50, marker='o')
    ax.scatter(test_class_1, np.ones(len(test_class_1)), alpha=0.6, color='green', 
              label=f'Test Class 1 ({len(test_class_1)} points)', s=50, marker='s')
    
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Class (0 or 1)')
    ax.set_title(f'Logistic Regression Dataset - Train/Test Split (Seed: {st.session_state.logistic_random_seed})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 1.5)
    
    st.pyplot(fig)
    
    # Train logistic regression model
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train_logistic, y_train_logistic)
    y_pred_logistic = logistic_model.predict(X_test_logistic)
    y_pred_proba_logistic = logistic_model.predict_proba(X_test_logistic)[:, 1]  # Probability of class 1
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_logistic, y_pred_logistic)
    
    # Display model parameters
    st.subheader("2.3 Model Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Coefficient (Weight)", f"{logistic_model.coef_[0][0]:.3f}", help="How much the feature influences the prediction")
    with col2:
        st.metric("Intercept (Bias)", f"{logistic_model.intercept_[0]:.3f}", help="The baseline prediction when feature = 0")
    
    st.info(f"**Learned logistic equation:** P(Class=1) = 1 / (1 + e^(-({logistic_model.coef_[0][0]:.3f} × x + {logistic_model.intercept_[0]:.3f})))")
    
    # Visualize results
    st.subheader("2.3 Model Visualization")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Training data with S-curve
    # Plot training data points
    train_class_0 = X_train_logistic[y_train_logistic == 0]
    train_class_1 = X_train_logistic[y_train_logistic == 1]
    ax1.scatter(train_class_0, np.zeros_like(train_class_0), alpha=0.6, color='red', 
              label=f'Training Class 0 ({len(train_class_0)} points)', s=50, marker='o')
    ax1.scatter(train_class_1, np.ones(len(train_class_1)), alpha=0.6, color='red', 
              label=f'Training Class 1 ({len(train_class_1)} points)', s=50, marker='s')
    
    # Create S-curve (sigmoid function)
    X_line = np.linspace(X_logistic.min(), X_logistic.max(), 300).reshape(-1, 1)
    y_proba_line = logistic_model.predict_proba(X_line)[:, 1]  # Probability of class 1
    ax1.plot(X_line, y_proba_line, color='blue', linewidth=3, label='Logistic S-Curve')
    
    # Add decision boundary line at 0.5 probability
    ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
    
    ax1.set_xlabel('Feature Value')
    ax1.set_ylabel('Probability / Class')
    ax1.set_title('Training Data with Logistic S-Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot 2: Test data with S-curve
    # Plot test data points
    test_class_0 = X_test_logistic[y_test_logistic == 0]
    test_class_1 = X_test_logistic[y_test_logistic == 1]
    ax2.scatter(test_class_0, np.zeros_like(test_class_0), alpha=0.6, color='green', 
              label=f'Test Class 0 ({len(test_class_0)} points)', s=50, marker='o')
    ax2.scatter(test_class_1, np.ones(len(test_class_1)), alpha=0.6, color='green', 
              label=f'Test Class 1 ({len(test_class_1)} points)', s=50, marker='s')
    
    # Add vertical lines to show prediction errors
    for i in range(len(y_test_logistic)):
        ax2.plot([X_test_logistic[i], X_test_logistic[i]], 
                [y_test_logistic[i], y_pred_proba_logistic[i]], 
                'r-', alpha=0.5, linewidth=1)
    
    # Same S-curve on test data
    ax2.plot(X_line, y_proba_line, color='blue', linewidth=3, label='Logistic S-Curve')
    
    # Add decision boundary line at 0.5 probability
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
    
    ax2.set_xlabel('Feature Value')
    ax2.set_ylabel('Probability / Class')
    ax2.set_title(f'Test Data with Logistic S-Curve (Accuracy: {accuracy:.3f})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.info(f"""
    **What you're seeing:**
    - **Left plot:** Training data with the learned S-shaped logistic curve
    - **Right plot:** Test data with the same S-curve to evaluate performance
    - **Blue S-curve:** Shows probability of belonging to Class 1 (ranges from 0 to 1)
    - **Decision boundary:** Points above 0.5 probability are classified as Class 1, below as Class 0
    - **Model accuracy:** {accuracy:.1%} of test predictions are correct
    """)
    
    # Create manual model copy for interactive adjustment
    logistic_model_manual = LogisticRegression()
    logistic_model_manual.fit(X_train_logistic, y_train_logistic)  # Fit to get all necessary attributes
    
    # Add slider to adjust manual model parameters
    st.subheader("2.4 Fit your own S curve")
    col1, col2 = st.columns(2)
    with col1:
        manual_coef = st.slider("Manual Coefficient", -5.0, 5.0, 1.0, 0.1, key="manual_coef")
    with col2:
        manual_intercept = st.slider("Manual Intercept", -5.0, 5.0, 0.0, 0.1, key="manual_intercept")
    
    # Update manual model parameters
    logistic_model_manual.coef_ = np.array([[manual_coef]])
    logistic_model_manual.intercept_ = np.array([manual_intercept])
    
    # Visualize manual model on test data only
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # Plot test data points
    ax.scatter(test_class_0, np.zeros_like(test_class_0), alpha=0.6, color='green', 
              label=f'Test Class 0 ({len(test_class_0)} points)', s=50, marker='o')
    ax.scatter(test_class_1, np.ones(len(test_class_1)), alpha=0.6, color='green', 
              label=f'Test Class 1 ({len(test_class_1)} points)', s=50, marker='s')
    
    # Manual model S-curve
    y_proba_line_manual = logistic_model_manual.predict_proba(X_line)[:, 1]
    ax.plot(X_line, y_proba_line_manual, color='orange', linewidth=3, label='Manual S-Curve')
    
    # Add vertical lines to show prediction errors with manual model
    y_pred_proba_manual = logistic_model_manual.predict_proba(X_test_logistic)[:, 1]
    for i in range(len(y_test_logistic)):
        ax.plot([X_test_logistic[i], X_test_logistic[i]], 
                [y_test_logistic[i], y_pred_proba_manual[i]], 
                'r-', alpha=0.5, linewidth=1)
    
    # Add decision boundary line at 0.5 probability
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision Boundary (0.5)')
    
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Probability / Class')
    ax.set_title(f'Test Data with Manual S-Curve (Coef: {manual_coef:.2f}, Intercept: {manual_intercept:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    st.pyplot(fig2)
    
    # Calculate manual model accuracy
    y_pred_manual = logistic_model_manual.predict(X_test_logistic)
    manual_accuracy = accuracy_score(y_test_logistic, y_pred_manual)
    
    st.info(f"""
    **Interactive Model:**
    - **Orange S-curve:** Your manually adjusted logistic curve
    - **Red lines:** Show prediction errors with your parameters
    - **Manual model accuracy:** {manual_accuracy:.1%} of test predictions are correct
    - Adjust the sliders to see how coefficient and intercept affect the curve shape and accuracy
    """)
