    
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def nlog_like(pdf, cdf, y_value, scale, indicator):
    
    u = np.log(pdf/(scale * np.exp(y_value)))
    c = np.log(1.0 - np.array(cdf))
    l = -(indicator * u + (1.0 - indicator) * c)
    
    return(l)



