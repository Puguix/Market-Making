from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


# %%%%%%%%%%%%%%%% RANDOM GENERATOR GLOBAL %%%%%%%%%%%%%%%%%
class RandomGenerator(ABC):

    """
    Random Generator Class for any probability distribution.
    """

    @abstractmethod
    def generate(self, *args, **kwargs):
        """
        Abstract method to generate 1 sample of a random number.
        """
        pass

# %%%%%%%%%%%%%%%% INTENSITY OF ORDERS ARRIVAL %%%%%%%%%%%%%%%%%
@dataclass
class ArrivalIntensity:
    
    """
    DataClass representing an order arrival intensity.

    Attributes:
        spread (float>0): the order book spread (mid to reference point).
        alpha (float>0): exponential parameter
        lambda_0 (float>0): multiplier parameter
    """

    spread:float = field(default=0.01)
    alpha:float = field(default=0.1)
    lambda_0:float = field(default=0.5)

    @property
    def intensity(self):
        return self.lambda_0 * np.exp(-self.alpha*self.spread)


# %%%%%%%%%%%%%%%% POISSON GENERATOR %%%%%%%%%%%%%%%%%
class PoissonGenerator(RandomGenerator):

    """
    A class that implement Poisson Random Number generation from a specified Arrival Intensity.

    Attributes:
        _lambda: the arrival intensity of order.
    """

    __slots__ = ["_lambda"]

    def __init__(self, arrival_intensity: ArrivalIntensity):

        self._lambda = arrival_intensity.intensity

    def generate(self)->int:
        
        draw = np.random.poisson(self._lambda)
        return draw
    
# %%%%%%%%%%%%%%%% TEST %%%%%%%%%%%%%%%%%
if __name__ =="__main__":

    gen_poisson = PoissonGenerator(ArrivalIntensity(0, 0, 5))
    print(gen_poisson.generate())