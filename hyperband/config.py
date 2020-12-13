

class HBConfig:

    def __init__(self, max_iter=81, eta=3):
        self.max_iter = max_iter
        self.eta = eta 

		self.logeta = lambda x: log( x ) / log( self.eta )
		self.s_max = int( self.logeta( self.max_iter ))
		self.B = ( self.s_max + 1 ) * self.max_iter