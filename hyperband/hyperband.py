import numpy as np 

from math import ceil
from time import time, ctime
from config import HBConfig


class Hyperband:

    def __init__(self, sample_params, evaluate_params, config: HBConfig):
        """Implement the efficient hyperband algorithm for hyper-parameter optimization

        Args:
            sample_params (function): callable, a function that can generate a set of hyper-parameters
            evaluate_params (function): callable, an evaluation function that return a dict including 'loss' 
                given a set of hyper-parameters and running budgets
            config (HBConfig): a configuration class for the hyperband algorithm
        """

        self.sample_params = sample_params
        self.evaluate_params = evaluate_params

        self.config = config

        self.results = []
        self.eval_counter = 0
        self.best_loss = np.inf 
        self.best_counter = -1

    def run(self):

        for s in range(self.config.s_max, -1, -1):

            n = int(ceil((self.s_max + 1) / (s + 1) * self.config.eta ** s))

            r = self.config.max_iter * self.config.eta ** (-s)

            n_params = [self.sample_params() for i in range(n)]

            for i in range(s + 1):

                ni = int(n * self.config.eta ** (-i))
                ri = r * self.config.eta ** i 

                print("\n *** {} configurations x {:.1f} iterations in each round". format(ni, ri))

                val_losses = []
                early_stops = []

                for p in n_params:
                    self.eval_counter += 1
                    print("\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format( 
						self.counter, ctime(), self.best_loss, self.best_counter ))

                    t_start = time()

                    # evaluate at given p up to ni iterations
                    result = self.evaluate_params(ni, p)

                    assert type(result) == dict 
                    assert 'loss' in result

                    eval_time_in_seconds = int(round(time() - t_start))

                    loss = result['loss']
                    val_losses.append( loss )
                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    # keep track of the best result so far 
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.eval_counter

                    result['counter'] = self.eval_counter
                    result['seconds'] = eval_time_in_seconds
                    result['params'] = p 
                    result['iterations'] = ni 

                    self.results.append(result)
                
                # select top-K configurations
                indices = np.argsort(val_losses)
                n_params = [n_params[i] for i in indices if not early_stops[i]]
                n_params = n_params[: int(ni / self.config.eta)]
                
        return self.results