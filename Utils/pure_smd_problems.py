import numpy as np

def smd1(ul_dim=10, ll_dim=10):
    
    problem_name = 'smd1' 
    r = ul_dim // 2
    p = ul_dim - r
    q = ll_dim - r
    
    ul_lb = -5*np.ones(ul_dim)       
    ul_ub = 10*np.ones(ul_dim)    

    eps = 1e-5
    ll_lb = np.concatenate([-5*np.ones(q), -np.pi/2*np.ones(r)+eps])
    ll_ub = np.concatenate([10*np.ones(q),  np.pi/2*np.ones(r)-eps])

    return ul_lb, ul_ub, ll_lb, ll_ub 


def smd2(ul_dim=10, ll_dim=10):
    
    problem_name = 'smd2'
    r = ul_dim // 2
    p = ul_dim - r
    q = ll_dim - r

    ul_lb = -5*np.ones(ul_dim)       
    ul_ub = np.concatenate([10*np.ones(p), np.ones(r)])

    eps = 1e-5
    ll_lb = np.concatenate([-5*np.ones(q), eps*np.ones(r)])
    ll_ub = np.concatenate([10*np.ones(q),  np.exp(1)*np.ones(r)])

    return ul_lb, ul_ub, ll_lb, ll_ub

def smd3(ul_dim=10, ll_dim=10):
    
    problem_name = 'smd3'
    r = ul_dim // 2
    p = ul_dim - r
    q = ll_dim - r

    ul_lb = -5*np.ones(ul_dim)       
    ul_ub = 10*np.ones(ul_dim)  

    eps = 1e-5
    ll_lb = np.concatenate([-5*np.ones(q), -np.pi/2*np.ones(r)+eps])
    ll_ub = np.concatenate([10*np.ones(q),  np.pi/2*np.ones(r)-eps])

    return ul_lb, ul_ub, ll_lb, ll_ub 

def smd4(ul_dim=10, ll_dim=10):
    
    problem_name = 'smd4'
    r = ul_dim // 2
    p = ul_dim - r
    q = ll_dim - r

    ul_lb = np.concatenate([-5*np.ones(p), -np.ones(r)])
    ul_ub = np.concatenate([10*np.ones(p),  np.ones(r)])

    ll_lb = np.concatenate([-5*np.ones(q), np.zeros(r)])
    ll_ub = np.concatenate([10*np.ones(q), np.exp(1)*np.ones(r)])

    return ul_lb, ul_ub, ll_lb, ll_ub 

def smd5(ul_dim=10, ll_dim=10):
    
    problem_name = 'smd5'

    ul_lb = -5*np.ones(ul_dim)       
    ul_ub = 10*np.ones(ul_dim)    

    ll_lb = -5*np.ones(ll_dim)       
    ll_ub = 10*np.ones(ll_dim)

    return ul_lb, ul_ub, ll_lb, ll_ub 


def smd6(ul_dim=10, ll_dim=10):
    
    problem_name = 'smd6'

    ul_lb = -5*np.ones(ul_dim)       
    ul_ub = 10*np.ones(ul_dim)    

    ll_lb = -5*np.ones(ll_dim)       
    ll_ub = 10*np.ones(ll_dim)

    return ul_lb, ul_ub, ll_lb, ll_ub 


def smd7(ul_dim=10, ll_dim=10):
    
    problem_name = 'smd7'
    r = ul_dim // 2
    p = ul_dim - r
    q = ll_dim - r
    
    ul_lb = -5*np.ones(ul_dim)
    ul_ub = np.concatenate([10*np.ones(p), np.ones(r)])

    eps = 1e-5
    ll_lb = np.concatenate([-5*np.ones(q), eps*np.ones(r)])
    ll_ub = np.concatenate([10*np.ones(q), np.exp(1)*np.ones(r)])

    return ul_lb, ul_ub, ll_lb, ll_ub 


def smd8(ul_dim=10, ll_dim=10):
    
    problem_name = 'smd8'

    ul_lb = -5*np.ones(ul_dim)       
    ul_ub = 10*np.ones(ul_dim)    

    ll_lb = -5*np.ones(ll_dim)       
    ll_ub = 10*np.ones(ll_dim)

    return ul_lb, ul_ub, ll_lb, ll_ub