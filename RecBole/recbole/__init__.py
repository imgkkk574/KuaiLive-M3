from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

__version__ = '1.0.1'

# Compatibility shim: restore scipy.sparse.dok_matrix._update (removed in scipy>=1.8).
# Base-RecBole models (NGCF, LightGCN, GCMC, SpectralCF) call A._update(data_dict) to
# bulk-set {(i,j): v} entries. Newer scipy (1.8+) deleted _update AND disabled dict-style
# .update() (NotImplementedError in 1.12+). Reimplement _update via item assignment,
# which works across all scipy versions. Mirrors the shim in recbole_cdr/__init__.py.
try:
    import scipy.sparse as _sp
    if not hasattr(_sp.dok_matrix, "_update"):
        def _dok_update(self, data_dict):
            for (i, j), v in data_dict.items():
                self[i, j] = v
        _sp.dok_matrix._update = _dok_update
except Exception:
    pass

# Compatibility shim: PyTorch 2.6+ changed torch.load's weights_only default from
# False to True. RecBole checkpoints pickle non-tensor objects (CDRConfig, etc.),
# so weights_only=True raises UnpicklingError at load time. Restore the old default
# (False) so all torch.load call sites in RecBole/RecBole-CDR load normally.
# Checkpoints here are our own (trusted), so the reduced safety posture is fine.
try:
    import torch as _torch
    _orig_load = _torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    _torch.load = _patched_load
except Exception:
    pass

# Compatibility shim: numpy 1.24+ removed the deprecated aliases np.float/np.int/
# np.bool/np.long/np.object/np.str/np.complex. RecBole 1.0.1 uses them throughout
# (metrics.py, layers.py, abstract_recommender.py). recbole_cdr has a
# compatibility_settings() that patches them, but it only runs via CDRConfig.__init__;
# load_data_and_model (checkpoint re-eval) reuses a pickled config and skips __init__,
# so the patch never applies there. Patch here at import time so every code path works.
try:
    import numpy as _np
    for _alias, _real in (("float", "float64"), ("int", "int_"), ("bool", "bool_"),
                          ("long", "int_"), ("object", "object_"), ("str", "str_"),
                          ("complex", "complex128"), ("unicode", "str_")):
        # use try/except instead of hasattr to avoid numpy 2.x FutureWarning on np.object/np.str
        try:
            getattr(_np, _alias)
        except AttributeError:
            setattr(_np, _alias, getattr(_np, _real))
except Exception:
    pass



