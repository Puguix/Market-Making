from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


# %%%%%%%%%%%%%%%% INTENSITY OF ORDERS ARRIVAL %%%%%%%%%%%%%%%%%
@dataclass
class ArrivalIntensity:

    spread:float = field(default=0.01)
    alpha:float = field(default=0.1)
    lambda_0:float = field(default=0.5)

    @property
    def intensity(self):
        return self.lambda_0 * np.exp(-self.alpha*self.spread)


# %%%%%%%%%%%%%%%% RANDOM GENERATOR GLOBAL %%%%%%%%%%%%%%%%%
class RandomGenerator(ABC):

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

# %%%%%%%%%%%%%%%% POISSON GENERATOR %%%%%%%%%%%%%%%%%
class PoissonGenerator(RandomGenerator):

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