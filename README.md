# Predictive Model for Crop Cultivation
### Project by:  
**Dinesh Kumar JB - 23MIA1126**  
**Aakash R - 23MIA1011**

---

## Overview

This project is a complete system designed to support crop cultivation decisions using machine learning. It focuses on two key tasks:

1. Predicting crop yield using regression models.
2. Recommending the most suitable crop for a given state and season.

It aims to provide valuable insights for agricultural planning based on historical data, environmental conditions, and resource usage.

---

## Features

### Yield Prediction
- It implements multiple regression models, including Decision Tree, Random Forest, AdaBoost, XGBoost, and LSTM.
- Evaluates each model using standard metrics such as Mean Squared Error (MSE) and R² score.
- Automatically selects and highlights the best-performing model.
- Saves results in CSV format and visualizes performance.

### Crop Recommendation
- Identifies the best crop for each combination of state and season.
- Uses a scoring system based on historical yield and resource efficiency.
- Trains a classification model to recommend crops based on user input.

---

## Tech Stack

| Component     | Technologies Used                        |
|---------------|-------------------------------------------|
| Language      | Python                                    |
| Libraries     | pandas, scikit-learn, xgboost, tensorflow, seaborn, matplotlib |
| Models        | Decision Tree, Random Forest, AdaBoost, XGBoost, LSTM |
| Interface     | Command-line based                        |

---

## Installation and Setup

1. Clone the repository

```bash
git clone https://github.com/yourusername/CropPredictor.git
cd CropPredictor
```

2. Install required packages

```bash
pip install -r requirements.txt
```

3. Run the modules

To predict crop yield:
```bash
python models/crop_yield_regression.py
```

To get a crop recommendation:
```bash
python models/crop_recommendation_classifier.py
```

---

## Example Output

### Yield Prediction
```
=====  Model Comparison Table (Sorted by R²) =====
                           MSE        R2  Rank
Random Forest     73583.050637  0.908164     1
Voting Ensemble   86628.148136  0.891882     2
XGBoost          104144.569394  0.870021     3
Decision Tree    105256.443520  0.868633     4
AdaBoost         106620.989445  0.866930     5
LSTM             408039.706196  0.490740     6
```

The model comparison results are saved as `model_comparison_regression.csv`.

### Crop Recommendation
```
--- Crop Suggestion System ---
Enter your State: tamil nadu
Enter the Season: kharif

Suggested Crop for 'Tamil Nadu' in 'Kharif': Jute
Based on high yield, low pesticide and fertilizer usage.
```

---

## Data Handling and Assumptions

- Handles missing values by replacing them with median (for numerical) or mode (for categorical).
- Normalizes text inputs for consistency.
- Label encodes state, season, and crop names for machine learning.
- Makes sure the prediction system handles real-world data inputs correctly.

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a new branch (e.g., `feature/new-feature`)
3. Commit your changes
4. Push the branch
5. Open a pull request

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or suggestions:

- **Email:** dineshkumar.j.b2005@gmail.com  
- **GitHub:** [Dineshkumar-jb](https://github.com/Dineshkumar-jb)  

---
