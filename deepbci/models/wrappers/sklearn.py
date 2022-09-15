from joblib import dump, load
from pathlib import Path
from pdb import set_trace

from deepbci.models.wrappers.base import BaseWrapper

class Sklearn(BaseWrapper):
    
    def __init__(self, model=None, load=None):
        super().__init__()
        
        if load:
            self.load(**load)
        elif model:
            self.model = model
        else:
            err = "Neither model nor load arguments were passed to initialize the Sklearn wrapper model."
            raise ValueError(err)      

    def fit(self, trn_set, vld_set=None, **kwargs):
        X, y = trn_set[0], trn_set[1]
        self.model.fit(X, y, **kwargs)

    def predict(self, tst_set, return_probs=False, **kwargs):
        X, y = tst_set[0], tst_set[1]
        if return_probs:
            return self.model.predict_proba(X, **kwargs)
        return self.model.predict(X, **kwargs)

    def save(self, filepath):
        if not isinstance(filepath, Path):
            filepath = Path(filepath)
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)
        dump(self.model, filepath)
    
    def load(self, filepath):
        self.model = load(filepath)



