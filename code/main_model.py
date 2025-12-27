import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data (path from dataset_info.txt)
df = pd.read_csv("your_dataset.csv")  # replace with actual file name

feature_cols = [  # 13 numeric features used in report
    "activepowerraw", "activepowercalculatedbyconverter",
    "reactivepower", "reacticepowercalculatedbyconverter",
    "gridpower10minaverage",
    "windspeedraw", "winddirectionraw", "windspeedturbulence",
    "ambienttemperature",
    "generatorspeed", "generatorwindingtempmax",
    "nc1insidetemp", "nacelletemp",
]
X = df[feature_cols]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = GradientBoostingRegressor(
    n_estimators=120, learning_rate=0.08, max_depth=3,
    subsample=0.8, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print("Test RMSE:", rmse)
print("Test R²:", r2)
