#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 

def lorentz_resolution_seq( R, sigma, b, x_0, y_0, z_0, N, T_c, step):
    '''
    Inputs :  R, sigma, b the paramaters of the lorentz system 
              x0, y0, z0 the initial conditions
              N order of the taylor series development
              step the step size between two stages
    Outputs : vectors x,y,z containing all the trajectory of the particle
    
    '''
    # initialisation :  create empty array
    n_steps = int(T_c / step )+1
    x = np.zeros(n_steps + 1 )
    y = np.zeros(n_steps + 1)
    z = np.zeros(n_steps + 1)
    
    x[0] = x_0
    y[0] = y_0
    z[0] = z_0
    

    
    for t in range(1, (n_steps +1 )) : #loop over time
        
        #initialisation : store alpha1, beta1, gamma1
        alpha = np.zeros(N + 1)
        beta = np.zeros(N + 1)
        gamma = np.zeros(N + 1)
        
        alpha[0] = x[t-1]
        beta[0] = y[t-1]
        gamma[0] = z[t-1]
        
        alpha[1] = sigma * (-x[t-1] + y[t-1])
        beta[1] = R * x[t-1] - y[t-1] -x[t-1] * z[t-1]
        gamma[1] = x[t-1] * y[t-1] - b * z[t-1]
        
        for i in range(1, N ) : #loop over derivatives, [::-1] to invert the order of the elements of the vector
            
            div = 1/(i+1)
            alpha[i+1] = div * sigma * ( beta[i]  - alpha[i] )  
            beta[i+1] = div * ( R * alpha[i] - beta[i] - np.sum(alpha[:(i+1)][::-1] * gamma[:(i+1)]  ))
            gamma[i+1] = div * ( np.sum( alpha[:(i+1)][::-1] * beta[ :(i+1)]) - b * gamma[i]  )
        
        x[t] = x[t-1] + np.sum( alpha[1:] * step**np.arange(1,N+1) )
        y[t] = y[t-1] + np.sum( beta[1:] * step**np.arange(1,N+1) )
        z[t] = z[t-1] + np.sum( gamma[1:] * step**np.arange(1,N+1) )
        
    return(x,y,z)



from timeit import Timer

def mesure_time_dim(stmt, contexts, nom_fonction, repeat = 10, number = 50, verbose=0):
    for context in contexts:
        if "ordre_name" not in context:
            raise ValueError("Pas cette valeur")
        res = mesure_time(stmt, context, repeat=repeat, number=number, nom_fonction = nom_fonction)
        res["ordre_name"] = context["ordre_name"]
        yield res
        
def mesure_time(stmt, context, nom_fonction, repeat=10, number = 50):
    tim = Timer(stmt, globals=context, setup = "from __main__ import "+str(nom_fonction))
    res = np.array(tim.repeat(repeat=repeat, number=number))
    mean = np.mean(res)
    mes = dict(average=mean, min_exec = np.min(res), 
               max_exec = np.max(res), repeat=repeat, number=number)
    return mes


