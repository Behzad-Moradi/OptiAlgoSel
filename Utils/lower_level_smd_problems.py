import numpy as np

def ll_problem(problem_name, ul_solution, ll_solution):    
    problem_func = globals().get(problem_name)
    if problem_func is None:
        raise ValueError(f"Problem function '{problem_name}' is not defined.")
    return problem_func(ul_solution, ll_solution)

# --- Sample SMD Problems ---

def smd1(xu, xl):
    r = len(xu) // 2
    p = len(xu) - r
    q = len(xl) - r

    xu1, xu2 = xu[:p], xu[p:p+r]
    xl1, xl2 = xl[:q], xl[q:q+r]

    f_val = np.sum(xu1**2) + np.sum(xl1**2) + np.sum((xu2 - np.tan(xl2))**2)
    return f_val

def smd2(xu, xl):
    r = len(xu) // 2
    p = len(xu) - r
    q = len(xl) - r

    xu1, xu2 = xu[:p], xu[p:p+r]
    xl1, xl2 = xl[:q], xl[q:q+r]

    f_val = np.sum(xu1**2) + np.sum(xl1**2) + np.sum((xu2 - np.log(xl2))**2)
    return f_val

def smd3(xu, xl):
    r = len(xu) // 2
    p = len(xu) - r
    q = len(xl) - r

    xu1, xu2 = xu[:p], xu[p:p+r]
    xl1, xl2 = xl[:q], xl[q:q+r]

    f_val = np.sum(xu1**2) + q + np.sum(xl1**2 - np.cos(2 * np.pi * xl1))+ np.sum((xu2**2 - np.tan(xl2))**2)
    return f_val

def smd4(xu, xl):
    r = len(xu) // 2
    p = len(xu) - r
    q = len(xl) - r

    xu1, xu2 = xu[:p], xu[p:p+r]
    xl1, xl2 = xl[:q], xl[q:q+r]

    f_val = np.sum(xu1**2) + q + np.sum(xl1**2 - np.cos(2 * np.pi * xl1))+ np.sum((np.abs(xu2) - np.log1p(xl2))**2)
    return f_val

def smd5(xu, xl):
    r = len(xu) // 2
    p = len(xu) - r
    q = len(xl) - r

    xu1, xu2 = xu[:p], xu[p:p+r]
    xl1, xl2 = xl[:q], xl[q:q+r]

    term = 0
    for i in range(q-1):
        term += (xl1[i+1] - xl1[i]**2)**2 + (xl1[i] - 1)**2

    f_val = np.sum(xu1**2) + term + np.sum((np.abs(xu2) - xl2**2)**2)
    return f_val

def smd6(xu, xl):
    r = len(xu) // 2
    p = len(xu) - r
    q = int(np.floor((len(xl) - r)/2 - np.finfo(float).eps))
    s = int(np.ceil((len(xl) - r)/2 + np.finfo(float).eps))

    xu1, xu2 = xu[:p], xu[p:p+r]
    xl1, xl2 = xl[:q+s], xl[q+s:q+s+r]

    term = np.sum(xl1[:q]**2)
    for i in range(q, q+s-1, 2):
        term += (xl1[i+1] - xl1[i])**2

    f_val = np.sum(xu1**2) + term + np.sum((xu2 - xl2)**2)
    return f_val

def smd7(xu, xl):
    r = len(xu) // 2
    p = len(xu) - r
    q = len(xl) - r

    xu1, xu2 = xu[:p], xu[p:p+r]
    xl1, xl2 = xl[:q], xl[q:q+r]

    f_val = np.sum(xu1**3) + np.sum(xl1**2) + np.sum((xu2 - np.log(xl2))**2)
    return f_val

def smd8(xu, xl):
    r = len(xu) // 2
    p = len(xu) - r
    q = len(xl) - r

    xu1, xu2 = xu[:p], xu[p:p+r]
    xl1, xl2 = xl[:q], xl[q:q+r]

    term = 0
    for i in range(q - 1):
        term += (xl1[i+1] - xl1[i]**2)**2 + (xl1[i]-1)**2

    f_val = np.sum(np.abs(xu1)) + term + np.sum((xu2 - xl2**3)**2)
    return f_val
