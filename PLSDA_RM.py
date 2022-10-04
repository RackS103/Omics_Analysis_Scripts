# Rac Mukkamala, White Lab
# PLS-DA scikit-learn plugin
# This PLSDA class is a subclass of sklearn, so it can be combined with all traditional sklearn pipelines
# and also has all of the traditional sklearn classifier functions.

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import TransformerMixin, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score


class PLSClassifier(TransformerMixin, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=self.n_components, max_iter=1000)
    
    def fit(self, X, Y):       
        enc = OneHotEncoder()
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        
        Y_proba = enc.fit_transform(np.reshape(Y,(-1,1))).toarray()
        self.labels = enc.categories_[0]
        self.pls.fit(X, Y_proba)
        
        self.x_weights_ = self.pls.x_weights_
        self.x_loadings_ = self.pls.x_loadings_
        self.x_rotations_ = self.pls.x_rotations_
        self.x_scores_ = self.pls.x_scores_
        self.y_weights_ = self.pls.y_weights_
        self.y_loadings_ = self.pls.y_loadings_
        self.y_rotations_ = self.pls.y_loadings_
        self.y_scores_ = self.pls.y_scores_
        self.coef_ = self.pls.coef_

        return self

    def transform(self, X):
        return self.pls.transform(X)

    def fit_transform(self, X, Y):
        return self.fit(X, Y).transform(X)

    def predict(self, X):
        pred_proba = self.pls.predict(X)
        idxs = np.argmax(pred_proba, axis=1)
        return np.reshape(self.labels[idxs], (-1,1))

    def score(self, X, Y):
        return accuracy_score(Y, self.predict(X))

    def set_params(self, **params):
        for a in params:
            if a == 'n_components':
                self.pls.set_params(n_components=params[a])

    def __repr__(self):
        return f'PLSClassifier(n_components={self.n_components})'

    def __str__(self):
        return repr(self)