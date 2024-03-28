import faiss

FAISS_MEASUREMENT_METRIC_DICT = {
    'cos': faiss.METRIC_INNER_PRODUCT,
    'l1': faiss.METRIC_L1,
    'l2': faiss.METRIC_L2,
    'l_inf': faiss.METRIC_Linf,
    'l_p': faiss.METRIC_Lp,
    'brayCurtis': faiss.METRIC_BrayCurtis,
    'canberra': faiss.METRIC_Canberra,
    'jensen_shannon': faiss.METRIC_JensenShannon
}


def get_measurement(measurement):
    return FAISS_MEASUREMENT_METRIC_DICT[measurement]
