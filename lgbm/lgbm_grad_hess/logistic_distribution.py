import numpy as np
import warnings
warnings.filterwarnings("ignore")



## log logistic (Z ~ logistic(0,1))

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

        
    

def logistic_pdf1(Z):
    exp = np.vectorize(np.exp)
    w = exp(Z)
    sqrt_denominator = 1 + w
    if (np.isinf(w) or np.isinf(w * w)):
        return (0.0)
    else:
        return w / (sqrt_denominator * sqrt_denominator)

logistic_pdf = np.vectorize(logistic_pdf1)    
           
def logistic_cdf1(Z):
    exp = np.vectorize(np.exp)
    w = exp(Z)
    if(np.isinf(w)):
        return(1.0)
    else:  
        return((w / (1 + w))) 
    
logistic_cdf = np.vectorize(logistic_cdf1)    


    
def logistic_grad1(Z):
    exp = np.vectorize(np.exp)
    w = exp(Z)
        
    if (np.isinf(w) or np.isinf(w * w)):
        return (0.0)
    else:
        return (logistic_pdf(Z) * ((1 - w) / (1 + w)))

logistic_grad = np.vectorize(logistic_grad1)    
        

def logistic_hess1(Z):
    exp = np.vectorize(np.exp)
    w = exp(Z)
    sqrt_denominator = 1 + w
        
    if (np.isinf(w) or np.isinf(w * w)):
        return (0.0)
    else:
        return (logistic_pdf(Z) * ((w * w - 4 * w + 1.0) / (sqrt_denominator * sqrt_denominator)))

logistic_hess = np.vectorize(logistic_hess1)    
        
     
    #@property
def logistic_gnumerator_u(Z):
    return(logistic_grad(Z))
    
    #@property
def logistic_gdenominator_u(Z, b):
    return(b * logistic_pdf(Z))
    
    #@property
def logistic_gnumerator_c(Z): 
    return(- logistic_pdf(Z))
    
    #@property
def logistic_gdenominator_c(Z, b): 
    return(b * (1.0 - logistic_cdf(Z)))
    
    #@property
def logistic_hnumerator_u(Z):
    return((logistic_grad(Z) * logistic_grad(Z)) - logistic_pdf(Z) * logistic_hess(Z))
        
    #@property
def logistic_hdenominator_u(Z, b):
    return((b * b * logistic_pdf(Z) * logistic_pdf(Z)))
    
    #@property
def logistic_hnumerator_c(Z):
    return((logistic_pdf(Z) * logistic_pdf(Z)) - ((1.0 - logistic_cdf(Z)) * - logistic_grad(Z)))
    
    #@property
def logistic_hdenominator_c(Z, b):
    return((b * b * (1.0 - logistic_cdf(Z))*(1.0 - logistic_cdf(Z))))


        
        
    


# Get gradient and hessian (log-logistic)

def logistic_getgrad1(z_value, sigma, grad_numerator_u, grad_denominator_u,
                        grad_numerator_c, grad_denominator_c):
    
    z_sign = z_value > 0.0 
    
    grad_u = zero_division(num = grad_numerator_u, deno = grad_denominator_u)
    grad_c = zero_division(num = grad_numerator_c, deno = grad_denominator_c)

        
    if(grad_denominator_u < kEps or np.isinf(grad_u) or np.isnan(grad_u)):
        if (z_sign):
            grad_u = 1/sigma
        else:
            grad_u = -1/sigma
                    
    if(grad_denominator_c < kEps or np.isinf(grad_c) or np.isnan(grad_c)):
        if (z_sign):
            grad_c = 0.0
        else:
            grad_c = -1/sigma   
                
    return(grad_u, grad_c)   



def logistic_gethess1(z_value, sigma, hess_numerator_u, hess_denominator_u,
                        hess_numerator_c, hess_denominator_c):
    
    z_sign = z_value > 0.0 
    hess_u = zero_division(num = hess_numerator_u, deno = hess_denominator_u)
    hess_c = zero_division(num = hess_numerator_c, deno = hess_denominator_c)


    if(hess_denominator_u < kEps or np.isinf(hess_u) or np.isnan(hess_u)):
        if (z_sign):
            hess_u = kMinHessian
        else:
            hess_u = kMinHessian
                    
    if(hess_denominator_c < kEps or np.isinf(hess_c) or np.isnan(hess_c)):
        if (z_sign):
            hess_c = kMinHessian
        else:
            hess_c = kMinHessian  
                
    return(hess_u, hess_c)   

    
logistic_getgrad = np.vectorize(logistic_getgrad1)
logistic_gethess = np.vectorize(logistic_gethess1)



