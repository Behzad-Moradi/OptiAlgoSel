import numpy as np
from torch.quasirandom import SobolEngine
from pflacco.sampling import create_initial_sample

def initial_sample_gen(dim = 2, n = 'None', sample_coefficient = 50, lower_bound = 0, upper_bound = 1, scramble=True, sample_type = 'lhs', seed ='None'):
    
    assert sample_type in ['random', 'lhs', 'sobol'], 'sample_type must be one of random, lhs, or sobol'
    if n == 'None':
        n = 50*dim  
    if sample_type in ['random', 'lhs']:
        X = create_initial_sample(dim = dim, n = n, lower_bound = lower_bound, upper_bound = upper_bound, sample_type = sample_type, seed = seed).values
    elif sample_type == 'sobol':
        soboleng = SobolEngine(dimension = dim, scramble = scramble, seed = int(seed))
        X = (upper_bound-lower_bound)*np.array(soboleng.draw(n))+lower_bound   
    return X
    