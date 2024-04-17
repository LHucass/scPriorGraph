from rpy2.robjects import r
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()


def get_mat_path(sc_path, pa_path, output_path):
    importr("GSEABase")
    importr("AUCell")
    importr("SingleCellExperiment")
    print('R package load!')
    r.source('./AucForPy2.R')
    r.AUC(sc_path, pa_path, output_path)

