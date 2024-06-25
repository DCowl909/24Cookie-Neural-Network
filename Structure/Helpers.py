import math


def sig(x: float) -> float:
    C = math.e
    return 1/(1+C**(-x))\
        
