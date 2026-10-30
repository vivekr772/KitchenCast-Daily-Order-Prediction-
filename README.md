# 🍳 KitchenCast: Daily Order Prediction Dashboard
### *Cloud Kitchen Food Delivery Demand Forecasting Using Artificial Neural Networks*

An interactive analytics dashboard and educational machine learning application built for the **MBA Data Science & AI** subject project in *"Neural Networks and Deep Learning"*.

---

## 📌 Project Overview
Cloud kitchen businesses (operating exclusively via aggregators like Swiggy, Zomato, and Direct D2C apps) face a critical operational dilemma:
- **Under-prediction (Stockouts):** Lost revenue, cancelled orders, and algorithmic search demotion on food apps.
- **Over-prediction (Food Waste):** Perishable raw material spoilage and excessive inventory holding costs.

**KitchenCast** provides an end-to-end client-side demand forecasting platform. It synthesizes realistic multi-platform operational logs, standardizes features, trains both a **Linear Regression baseline** and a **Deep Feedforward Artificial Neural Network (MLP)** directly in the browser via **TensorFlow.js**, evaluates them on a strictly held-out test set, and enables operations managers to simulate "What-If" business decisions (e.g., rainfall shocks, promotional discount changes, ad spend scaling).

---

## 🚀 Key Features

1. **Executive Overview Dashboard**: Real-time KPI summary cards (Total Records, Volume, Revenue, Discounts, Delivery Time, Best-Selling Cuisine) paired with interactive trend, cuisine distribution, channel share, weekday seasonality, and operational friction charts.
2. **Synthetic Data Synthesizer**: Generates 1,000 to 3,000 multi-attribute records with realistic econometric domain logic (cuisine base orders, platform multipliers, weekend/festival uplifts, monsoon rainfall shocks, non-linear ad spend diminishing returns, and delivery time penalties).
3. **In-Browser Deep Learning with TensorFlow.js**:
   - **Baseline Model:** Ordinary Least Squares Linear Regression (`Dense(1, linear)`).
   - **ANN Model:** Multilayer Perceptron (`Dense(32, ReLU) → Dropout(0.2) → Dense(16, ReLU) → Dense(1, linear)`).
   - **Adam Optimizer & MSE Loss Function**.
   - **Early Stopping:** Tracks validation loss with a patience of 10 epochs to prevent overfitting.
   - **Real-Time Convergence Charting:** Live updates of training and validation loss curves.
4. **Held-Out Test Set Evaluation**:
   - Evaluates on a strictly unseen 20% test partition (no data leakage).
   - Reports **MAE (Mean Absolute Error)**, **RMSE (Root Mean Squared Error)**, and **$R^2$ (Coefficient of Determination)**.
   - Actual vs. Predicted scatter plot with perfect $y = x$ 45° reference line.
   - Residual error distribution histogram.
   - Automated honest executive text insight explaining the operational difference.
5. **Operational What-If Predictor & Scenario Simulator**:
   - Interactive control panel to test menu pricing, aggregator choice, festival dates, and weather conditions.
   - **Promotional Elasticity Curve:** Plots real neural network predictions across discount levels from 0% to 50% to identify the revenue-maximizing discount threshold.
6. **Academic & Viva Reference Tab**: Comprehensive, textbook-grounded explanations of neural network fundamentals designed specifically for MBA viva examinations.

---

## 🛠️ Technology Stack
- **Architecture:** Zero-backend, 100% standalone single-file application (`index.html`).
- **Machine Learning Engine:** [TensorFlow.js](https://www.tensorflow.org/js) (v4.17.0 via CDN).
- **Interactive Visualizations:** [Chart.js](https://www.chartjs.org/) (v4.4.1 via CDN).
- **Styling:** Custom modern executive dashboard design (Saffron Orange & Dark Charcoal palette, responsive CSS Grid / Flexbox).
- **Fonts:** Inter & JetBrains Mono (Google Fonts).

---

## 🏃 How to Run the Application

No Python, Node.js, database, or server installation is required.

1. Double-click or open `index.html` in any modern web browser (Google Chrome, Microsoft Edge, Mozilla Firefox, or Safari).
2. *Note:* An internet connection is required only to load the TensorFlow.js and Chart.js CDNs on initial page load.

---

## 🧠 Model Architecture & Hyperparameters

| Component | Baseline: Linear Regression | Deep Artificial Neural Network (ANN) |
|---|---|---|
| **Architecture** | `Dense(1)` | `Dense(32, ReLU) → Dropout(0.2) → Dense(16, ReLU) → Dense(1, linear)` |
| **Input Dimension** | 25 features (one-hot & standardized) | 25 features (one-hot & standardized) |
| **Total Parameters** | 26 parameters | 1,361 trainable parameters |
| **Activation (Hidden)** | None (Linear) | ReLU ($\max(0, z)$) |
| **Activation (Output)** | Linear (Continuous demand) | Linear (Continuous demand) |
| **Regularization** | None | Dropout (20%) + Early Stopping (Patience = 10) |
| **Optimizer** | Adam ($\text{lr} = 0.001$) | Adam ($\text{lr} = 0.001$) |
| **Loss Function** | Mean Squared Error (MSE) | Mean Squared Error (MSE) |
| **Data Partitioning** | 80% Train (85% Fit / 15% Val) / 20% Test | 80% Train (85% Fit / 15% Val) / 20% Test |

---

## 🎓 MBA Viva Voce Quick Reference

### 1. Why use an ANN for order prediction instead of simple linear regression?
> *Kitchen demand exhibits non-linear interactions:* e.g., marketing spend yields diminishing returns ($\ln(1 + \text{spend})$), rainy weather amplifies discount sensitivity, and delivery delays compound order cancellations. While linear regression only estimates flat additive slopes, ANNs use multi-layer non-linear activations (ReLU) to map complex, non-linear multi-variable interactions.

### 2. Why is the output layer Linear and not Sigmoid or Softmax?
> Sigmoid bounds outputs between 0 and 1 (ideal for binary classification/probabilities), and Softmax normalizes outputs into probability distributions summing to 1 (multiclass classification). Daily order demand is an **unbounded continuous regression target**, requiring a **linear identity activation** ($y = w \cdot x + b$).

### 3. How do you prevent the neural network from overfitting?
> Three protective mechanisms are applied:
> 1. **Dropout (0.2):** Randomly deactivates 20% of hidden neurons during each forward pass, forcing the network to learn redundant, robust features rather than memorizing noise.
> 2. **Validation Split & Early Stopping:** A 15% validation split monitors generalization error each epoch; training stops if validation loss fails to improve for 10 epochs.
> 3. **Z-Score Normalization on Training Set Only:** Feature means and standard deviations are computed strictly on the training partition and then applied to validation/test sets to eliminate data leakage.

### 4. What are the key evaluation metrics?
> - **MAE (Mean Absolute Error):** Average absolute forecast error in actual kitchen order units (e.g., off by ~4 orders per day). Directly understandable by operational staff.
> - **RMSE (Root Mean Squared Error):** Heavily penalizes large forecast outliers due to squaring.
> - **$R^2$ (Coefficient of Determination):** Percentage of total daily order variance explained by the model compared to predicting the mean.

---

## 📄 License & Attribution
Developed for educational and academic demonstration purposes for the MBA Data Science & Artificial Intelligence curriculum.
