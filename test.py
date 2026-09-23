import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from foresight_gpu import GPURegressor
from foresight_gpu.models import MLPModel
from foresight_gpu.scoring import reliability_scorer, make_gpu_scorer
import matplotlib

matplotlib.use('qtagg')
rng = np.random.default_rng(0)
X = rng.uniform(-1, 1, size=(500, 2))
y = np.sin(2*np.pi*X[:, 0]) + 0.3*X[:, 1] + 0.25*rng.standard_normal(500)
cut = 400
X_tr, y_tr, X_va, y_va = X[:cut], y[:cut], X[cut:], y[cut:]
print('train', X_tr.shape, '| validation', X_va.shape)
demo = GPURegressor(population=10, metric='mse', n_iter=200,
                    hv_penalty=10, random_state=0).fit(X_tr, y_tr)
ens = demo.ensemble_

P = demo.hv_penalty_
eta, loss, front = ens.front_objectives(X_va, y_va)
e, l = eta[front], np.clip(loss[front], 0.0, P)

parts = ens.score_hypervolume(X_va, y_va, details=True)


width = np.diff(e)
height = np.maximum(l[:-1], l[1:])


fig, ax = plt.subplots(figsize=(6, 4))
plt.show(block=False)
ax.plot(e, l, 'o', label='front')
ax.set_xlabel('Eta')
ax.set_ylabel('Loss')
ax.legend()
plt.show(block=False)