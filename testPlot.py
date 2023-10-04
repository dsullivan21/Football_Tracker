def f(x):
    """The function to predict."""
    return x * np.sin(x)


rng = np.random.RandomState(42)

arr = [174,251,340,283,175,182,446,76,273,273,321,315,374,443,378,278,254,393,318,462,348,372,416,200,225,340,236,385,302,211,270,
259,258,410,258,184,260,406,166,275,206,397,272,278,260,343,337,202,328,224,336,352,223,320,329,331,446,423,338,292,249,262,235,360,
202,328,224,336,352,223,320,329,331,446,423,338,292,249,262,235,360,272,305,226]

i = 0

while i < len(arr):
    arr[i] = arr[i]/100
    i = i + 1

X = np.atleast_2d(arr)



expected_y = X.ravel()

print("X: " , X)
print("Y: " , expected_y)

X = X.reshape(-1,1)

sigma = 0.5 + X.ravel() / 10
noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
y = expected_y + noise

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss, mean_squared_error

all_models = {}
common_params = dict(
    learning_rate=0.05,
    n_estimators=200,
    max_depth=2,
    min_samples_leaf=9,
    min_samples_split=9,
)

print(y_train)
for alpha in [0.15, 0.5, 0.85]:
    gbr = GradientBoostingRegressor(loss="quantile", alpha=alpha, **common_params)
    all_models["q %1.2f" % alpha] = gbr.fit(X_train, y_train)

gbr_ls = GradientBoostingRegressor(loss="squared_error", **common_params)
all_models["mse"] = gbr_ls.fit(X_train, y_train)

xx = np.atleast_2d(np.linspace(0, 10, 1000)).T
import matplotlib.pyplot as plt

y_pred = all_models["mse"].predict(xx)
y_lower = all_models["q 0.15"].predict(xx)
y_upper = all_models["q 0.85"].predict(xx)
y_med = all_models["q 0.50"].predict(xx)

fig = plt.figure(figsize=(10, 10))
plt.plot(xx, f(xx), "g:", linewidth=3, label=r"$f(x) = x\,\sin(x)$")
plt.plot(X_test, y_test, "b.", markersize=10, label="Test observations")
plt.plot(xx, y_med, "r-", label="Predicted median")
plt.plot(xx, y_pred, "r-", label="Predicted mean")
plt.plot(xx, y_upper, "k-")
plt.plot(xx, y_lower, "k-")
plt.fill_between(
    xx.ravel(), y_lower, y_upper, alpha=0.4, label="Predicted 70% interval"
)
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.ylim(0, 10)
plt.legend(loc="upper left")
plt.show()

