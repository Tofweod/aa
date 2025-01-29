from attacker import Attacker
import torch.nn as nn
import numpy as np
from ifgsm_utils import perturb_iterative


class get_attacker(Attacker):

    def __init__(self,predict,loss_fn,eps,nb_iter,
                 alpha,fixed=0.5,
                 rand_init=False,ord=np.inf,targeted=False) -> None:

        super(get_attacker,self).__init__(predict,loss_fn,fixed)
        self.eps = eps
        self.nb_iter = nb_iter
        self.alpha = alpha
        self.rand_init = rand_init
        
        self.ord = ord
        self.targeted = targeted

        self.fixed= fixed

        if self.loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss(reduction="sum")



    def perturb(self, x, y, **kwargs):

        # TODO: targeted

        rval = perturb_iterative(x,y,
                                 self.predict,self.loss_fn,
                                 kwargs['num_class'],self.nb_iter,self.alpha,self.eps,self.fixed)


        return rval
