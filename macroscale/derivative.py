import numpy as np

class Stencil:

    
    def central_difference(self, f: function, x: float, h: float) -> float:
        return (f(x + h) - f(x - h)) / (2 * h)