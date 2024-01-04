import numpy as np
import warnings
warnings.filterwarnings("ignore")


# Regularization

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


# Probability Distribution

## log normal distribution (Z ~ N(0,1))

### Normal
class normal_dist:
    
    kMinGradient = -15.0
    kMaxGradient = 15.0
    kMinHessian = 1e-16  # Ensure that no data point gets zero hessian
    kMaxHessian = 15.0 
    kEps = 1e-12;  # A denominator in a fraction should not be too small

    
    
    def __init__(self, Z, scale):
        self.Z = np.array(Z)
        self.b = np.array(scale)
        
    def normal_pdf(self):
        return(np.exp((-self.Z * self.Z)/2.0) / np.sqrt((2.0 * np.pi)))
        #normal_pdf = np.vectorize(normal_pdf1)


    def normal_cdf(self):
        import math
        math_erf = np.vectorize(math.erf) 
        return (0.5 * (1.0 + math_erf(self.Z / np.sqrt(2.0))))
        #normal_cdf = np.vectorize(normal_cdf1)

    def normal_grad(self):
        return(-self.Z * self.normal_pdf())
        #normal_grad = np.vectorize(normal_grad1)

    def normal_hess(self):
        return((self.Z * self.Z - 1.0) * self.normal_pdf())
        #normal_hess = np.vectorize(normal_hess1)   
    
    @property
    def gnumerator_u(self):
        return(self.normal_grad())
    
    @property
    def gdenominator_u(self):
        return(self.b * self.normal_pdf())
    
    @property
    def gnumerator_c(self): 
        return(- self.normal_pdf())
    
    @property
    def gdenominator_c(self): 
        return(self.b * (1.0 - self.normal_cdf()))
    
    @property
    def hnumerator_u(self):
        return((self.normal_grad() * self.normal_grad()) - self.normal_pdf() * self.normal_hess())
        
    @property
    def hdenominator_u(self):
        return((self.b*self.b * self.normal_pdf()*self.normal_pdf()))
    
    @property
    def hnumerator_c(self):
        return((self.normal_pdf() * self.normal_pdf()) - ((1.0 - self.normal_cdf()) * - self.normal_grad()))
    
    @property
    def hdenominator_c(self):
        return((self.b * self.b * (1.0 - self.normal_cdf())*(1.0 - self.normal_cdf())))

    
    
    
    ###########################################################################
    
    
    #############################################################################
    
    
   

# Get gradient and hessian

def normal_getgrad1(z_value, sigma, grad_numerator_u, grad_denominator_u,
                        grad_numerator_c, grad_denominator_c):
    
    z_sign = z_value > 0.0 
    
    grad_u = zero_division(num = grad_numerator_u, deno = grad_denominator_u)
    grad_c = zero_division(num = grad_numerator_c, deno = grad_denominator_c)

        
    if(grad_denominator_u < kEps or np.isinf(grad_u) or np.isnan(grad_u)):
        if (z_sign):
            grad_u = kMaxGradient
        else:
            grad_u = kMinGradient
                    
    if(grad_denominator_c < kEps or np.isinf(grad_c) or np.isnan(grad_c)):
        if (z_sign):
            grad_c = 0.0
        else:
            grad_c = -15   
                
    return(grad_u, grad_c)   



def normal_gethess1(z_value, sigma, hess_numerator_u, hess_denominator_u,
                        hess_numerator_c, hess_denominator_c):
    
    z_sign = z_value > 0.0 
    hess_u = zero_division(num = hess_numerator_u, deno = hess_denominator_u)
    hess_c = zero_division(num = hess_numerator_c, deno = hess_denominator_c)


    if(hess_denominator_u < kEps or np.isinf(hess_u) or np.isnan(hess_u)):
        if (z_sign):
            hess_u = 1/(sigma*sigma)
        else:
            hess_u = 1/(sigma*sigma)
                    
    if(hess_denominator_c < kEps or np.isinf(hess_c) or np.isnan(hess_c)):
        if (z_sign):
            hess_c = kMinHessian
        else:
            hess_c = 1/(sigma*sigma)  
                
    return(hess_u, hess_c)   

    
normal_getgrad = np.vectorize(normal_getgrad1)
normal_gethess = np.vectorize(normal_gethess1)
