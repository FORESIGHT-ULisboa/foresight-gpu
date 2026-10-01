"""GPURegressor: scikit-learn contract, prediction, pipelines, warm start."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from foresight_gpu import GPURegressor
from foresight_gpu.models import MLPModel


@pytest.fixture
def data(rng):
    X = rng.uniform(-1, 1, size=(250, 2))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.4 * X[:, 1] + 0.2 * rng.standard_normal(250)
    return X, y


def _fit(data, **kw):
    X, y = data
    params = dict(population=120, n_iter=25, random_state=0)
    params.update(kw)
    return GPURegressor(**params).fit(X, y), X, y


class TestSklearnContract:
    def test_get_set_params(self):
        gpu = GPURegressor(population=100)
        assert gpu.get_params()["population"] == 100
        gpu.set_params(n_iter=10, metric="kge")
        assert gpu.n_iter == 10 and gpu.metric == "kge"

    def test_clone_preserves_params(self):
        gpu = GPURegressor(population=77, metric="rmse")
        cloned = clone(gpu)
        assert cloned.get_params()["population"] == 77
        assert cloned.get_params()["metric"] == "rmse"

    def test_init_does_not_mutate(self, data):
        # __init__ must not touch args; model stays None until fit resolves it.
        gpu = GPURegressor()
        assert gpu.model is None
        _fit(data)
        assert gpu.model is None

    def test_fit_returns_self(self, data):
        gpu, X, y = _fit(data)
        assert isinstance(gpu, GPURegressor)

    def test_nested_params_route(self):
        gpu = GPURegressor(model=MLPModel(n_hidden=8))
        assert gpu.get_params()["model__n_hidden"] == 8
        gpu.set_params(model__n_hidden=4)
        assert gpu.model.n_hidden == 4


class TestPrediction:
    def test_predict_shape_and_finite(self, data):
        gpu, X, y = _fit(data)
        pred = gpu.predict(X)
        assert pred.shape == (X.shape[0],)
        assert np.all(np.isfinite(pred))

    def test_predict_quantiles_shape(self, data):
        gpu, X, y = _fit(data)
        bands = gpu.predict_quantiles(X, quantiles=[0.1, 0.5, 0.9])
        assert bands.shape == (X.shape[0], 3)

    def test_feature_count_checked(self, data):
        gpu, X, y = _fit(data)
        with pytest.raises(ValueError):
            gpu.predict(X[:, :1])

    def test_dataframe_output(self, data):
        pd = pytest.importorskip("pandas")
        gpu, X, y = _fit(data)
        Xdf = pd.DataFrame(X, columns=["a", "b"])
        out = gpu.predict_quantiles(Xdf, quantiles=[0.25, 0.75])
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == [0.25, 0.75]

    def test_score_is_finite(self, data):
        gpu, X, y = _fit(data)
        assert np.isfinite(gpu.score(X, y))


class TestPipeline:
    def test_in_pipeline(self, data):
        X, y = data
        pipe = Pipeline(
            [("scale", StandardScaler()),
             ("gpu", GPURegressor(population=100, n_iter=20, random_state=0))]
        )
        pipe.fit(X, y)
        assert pipe.predict(X).shape == (X.shape[0],)

    def test_pipeline_is_now_the_route_for_input_scaling(self, data):
        """0.5.0: the estimator scales nothing, so the scaler has to be the caller's.

        The pipeline's StandardScaler is the only thing standing between raw X and the
        model, and it must survive predict as well as fit.
        """
        X, y = data
        offset = X + np.array([500.0, -300.0])   # far outside the model's comfortable range
        pipe = Pipeline(
            [("scale", StandardScaler()),
             ("gpu", GPURegressor(population=120, n_iter=20, random_state=0))]
        ).fit(offset, y)
        assert np.isfinite(pipe.predict(offset)).all()
        assert not hasattr(pipe[-1], "x_scaler_")   # nothing scales inside the estimator

    def test_target_range_is_the_models_business(self, data):
        """y is out of Pipeline's reach, so it is expressed on the model instead."""
        from foresight_gpu.models import MLPModel

        X, y = data
        big = y * 50.0 + 2000.0
        pipe = Pipeline(
            [("scale", StandardScaler()),
             ("gpu", GPURegressor(
                 model=MLPModel(output_scale=big.std(), output_offset=big.mean()),
                 population=120, n_iter=20, random_state=0))]
        ).fit(X, big)
        pred = pipe.predict(X)
        assert np.isfinite(pred).all()
        assert abs(np.mean(pred) - big.mean()) < 0.5 * big.std()

    def test_gridsearch_reaches_the_output_range(self, data):
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        from foresight_gpu.models import MLPModel
        from foresight_gpu.scoring import make_gpu_scorer

        X, y = data
        search = GridSearchCV(
            GPURegressor(model=MLPModel(), population=60, n_iter=8, random_state=0),
            {"model__output_scale": [1.0, float(np.std(y))]},
            cv=TimeSeriesSplit(2), scoring=make_gpu_scorer("hypervolume"),
        ).fit(X, y)
        assert "model__output_scale" in search.best_params_


class TestWarmStart:
    def test_warm_start_runs_and_retains_population(self, data):
        X, y = data
        gpu = GPURegressor(population=100, n_iter=15, warm_start=True, random_state=0)
        gpu.fit(X, y)
        pop1 = gpu._population.copy()
        gpu.fit(X, y)  # continue
        assert gpu._population.shape == pop1.shape
        assert np.all(np.isfinite(gpu.predict(X)))


class TestRegularization:
    def test_regularized_fit_runs(self, data):
        gpu, X, y = _fit(data, model=MLPModel(reg_lambda=1e-3, reg_p=1))
        assert np.all(np.isfinite(gpu.predict(X)))


class TestReproducibility:
    def test_same_seed_same_prediction(self, data):
        X, y = data
        a = GPURegressor(population=100, n_iter=20, random_state=7).fit(X, y).predict(X)
        b = GPURegressor(population=100, n_iter=20, random_state=7).fit(X, y).predict(X)
        np.testing.assert_allclose(a, b)
