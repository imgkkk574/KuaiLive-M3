from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

__version__ = '0.1.0'

# Compatibility shim: restore scipy.sparse.dok_matrix._update (removed in scipy>=1.8).
# RecBole-CDR / RecBole models (BiTGCF, NGCF, LightGCN, GCMC, SpectralCF) call the
# old private A._update(data_dict) to bulk-set {(i,j): v} entries. Newer scipy (1.8+)
# deleted _update AND disabled dict-style .update() (raises NotImplementedError).
# Reimplement _update via item assignment (A[i,j]=v), which still works on all versions.
try:
    import scipy.sparse as _sp

    if not hasattr(_sp.dok_matrix, "_update"):
        def _dok_update(self, data_dict):
            for (i, j), v in data_dict.items():
                self[i, j] = v
        _sp.dok_matrix._update = _dok_update
except Exception:
    pass  # scipy not available at import time; models will surface the real error


