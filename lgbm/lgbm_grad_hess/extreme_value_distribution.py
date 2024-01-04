import numpy as np
import warnings
warnings.filterwarnings("ignore")


## Weibull (Z ~ EV(0,1))
kMinGradient = -15.0
kMaxGradient = 15.0
kMinHessian = 1e-16  # Ensure that no data point gets zero hessian
kMaxHessian = 15.0
kEps = 1e-12;  # A denominator in a fraction should not be too small

##### regularized function

def regularization1(t, t_max, t_min):
    
    if(t < t_min):
        return t_min
    if(t > t_max):
        return t_max
    
    return t

regularization = np.vectorize(regularization1)



########## Handelling zero division error (x/0)

def zero_division1(num, deno):
    
    try:
        result = num / deno
    except ZeroDivisionError:
        result = np.inf
     
    return(result)

zero_division = np.vectorize(zero_division1)




def extreme_pdf1(Z):
    exp = np.vectorize(np.exp)
    w = exp(Z)
    if(np.isinf(w)):
        return (0.0)
    else:
        return(w * np.exp(-w))
extreme_pdf = np.vectorize(extreme_pdf1)    
    
    
def extreme_cdf1(Z):
    exp = np.vectorize(np.exp)
    w = exp(Z)
    if(np.isinf(w)):
        return (1.0)
    else:
        return(1 - exp(-w))
extreme_cdf = np.vectorize(extreme_cdf1)    


def extreme_grad1(Z):
    exp = np.vectorize(np.exp)
    w = exp(Z)
    if(np.isinf(w)):
        return (0.0)
    else:
        return extreme_pdf(Z) * (1.0 - w)
extreme_grad = np.vectorize(extreme_grad1)    

    
def extreme_hess1(Z):
    exp = np.vectorize(np.exp)
    w = exp(Z)
    if(np.isinf(w)):
        return (0.0)
    else:
        return extreme_pdf(Z) * (1.0 - 3 * w + w * w)
extreme_hess = np.vectorize(extreme_hess1)    

    
    
    
#    @property
def extreme_gnumerator_u(Z):
    return(extreme_grad(Z))
    
#    @property
def extreme_gdenominator_u(Z, b):
    return(b * extreme_pdf(Z))
    
#    @property
def extreme_gnumerator_c(Z): 
    return(- extreme_pdf(Z))
    
#    @property
def extreme_gdenominator_c(Z, b): 
    return(b * (1.0 - extreme_cdf(Z)))
    
#    @property
def extreme_hnumerator_u(Z):
    return((extreme_grad(Z) * extreme_grad(Z)) - extreme_pdf(Z) * extreme_hess(Z))
        
#    @property
def extreme_hdenominator_u(Z, b):
    return((b*b * extreme_pdf(Z)*extreme_pdf(Z)))
    
#    @property
def extreme_hnumerator_c(Z):
    return((extreme_pdf(Z) * extreme_pdf(Z)) - ((1.0 - extreme_cdf(Z))* - extreme_grad(Z)))
    
#    @property
def extreme_hdenominator_c(Z, b):
    return((b*b * (1.0 - extreme_cdf(Z))*(1.0 - extreme_cdf(Z))))


        
        
        



    

kMaxHessian

def extreme_getgrad1(z_value, sigma, grad_numerator_u, grad_denominator_u,
                        grad_numerator_c, grad_denominator_c):
    
    z_sign = z_value > 0.0 
    
    grad_u = zero_division(num = grad_numerator_u, deno = grad_denominator_u)
    grad_c = zero_division(num = grad_numerator_c, deno = grad_denominator_c)

        
    if(grad_denominator_u < kEps or np.isinf(grad_u) or np.isnan(grad_u)):
        if (z_sign):
            grad_u = 1/sigma
        else:
            grad_u = kMinGradient
                    
    if(grad_denominator_c < kEps or np.isinf(grad_c) or np.isnan(grad_c)):
        if (z_sign):
            grad_c = 0.0
        else:
            grad_c = kMinGradient 
                
    return(grad_u, grad_c)   



def extreme_gethess1(z_value, sigma, hess_numerator_u, hess_denominator_u,
                        hess_numerator_c, hess_denominator_c):
    
    z_sign = z_value > 0.0 
    hess_u = zero_division(num = hess_numerator_u, deno = hess_denominator_u)
    hess_c = zero_division(num = hess_numerator_c, deno = hess_denominator_c)


    if(hess_denominator_u < kEps or np.isinf(hess_u) or np.isnan(hess_u)):
        if (z_sign):
            hess_u = kMinHessian
        else:
            hess_u = kMaxHessian
                    
    if(hess_denominator_c < kEps or np.isinf(hess_c) or np.isnan(hess_c)):
        if (z_sign):
            hess_c = kMinHessian
        else:
            hess_c = kMaxHessian  
                
    return(hess_u, hess_c)   

    
extreme_getgrad = np.vectorize(extreme_getgrad1)
extreme_gethess = np.vectorize(extreme_gethess1)


