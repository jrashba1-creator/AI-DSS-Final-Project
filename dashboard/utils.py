"""
Custom wrapper for log-transformed regression.
Must be imported before loading Alkalinity model.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

class LogTransformedRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper that applies log1p transform to target before training,
    and expm1 transform to predictions.
    """
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
    
    def fit(self, X, y):
        self.model_ = clone(self.base_estimator)
        y_log = np.log1p(y)
        self.model_.fit(X, y_log)
        return self
    
    def predict(self, X):
        y_log_pred = self.model_.predict(X)
        return np.expm1(y_log_pred)
    
    def get_params(self, deep=True):
        params = {'base_estimator': self.base_estimator}
        if deep and hasattr(self.base_estimator, 'get_params'):
            base_params = self.base_estimator.get_params(deep=True)
            for key, value in base_params.items():
                params[f'base_estimator__{key}'] = value
        return params
    
    def set_params(self, **params):
        base_params = {}
        for key, value in params.items():
            if key.startswith('base_estimator__'):
                base_params[key.replace('base_estimator__', '')] = value
            elif key == 'base_estimator':
                self.base_estimator = value
        
        if base_params and hasattr(self.base_estimator, 'set_params'):
            self.base_estimator.set_params(**base_params)
        return self
