

from abc import ABCMeta

class Attacker(object):

    __metaclass__ = ABCMeta

    def __init__(self,predict,loss_fn,fixed):

        self.predict = predict
        self.loss_fn = loss_fn
        self.fixed = fixed
        

    def perturb(self,x,y,**kwargs):

        error = "Subclass must implement perturb"
        raise NotImplementedError(error)


    def __call__(self,*args,**kwargs):
        return self.perturb(*args,**kwargs)
