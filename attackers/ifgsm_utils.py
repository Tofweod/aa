import torch
import pdb
import numpy as np

def batch_clip_y(x,y,fixed)->torch.Tensor:
    # val: BxDxN

    shape = x.size()
    fixed_num = shape[2] * fixed

    r_mat = torch.rand(shape)
    _,indices = torch.topk(r_mat,k=fixed_num,dim=2,largest=False)

    mask = torch.zeros_like(x,dtype=torch.int)
    mask.scatter(2,indices,1)

    y[~(mask == 1)] = 0

    # batch add
    return x + y


def batch_sign_mask(sign:torch.Tensor,pred,target):
    mask = (pred == target).view(-1,1,1).expand_as(sign)
    
    return sign.masked_fill(~mask,0)



def perturb_iterative(x,y,predict,loss_fn,num_class,nb_iter,alpha,eps,
                      fixed):
    #  pdb.set_trace()
    # x:BxDxN
    out_val = x
    pred = torch.zeros(y.size()[0],num_class,dtype=torch.float32)
    #  out_pred = y

    for _ in range(nb_iter):
        # TODO: all batch data eps break condition 
        diff = torch.norm((out_val-x),p=float("inf"),dim=(1,2))
        #  if torch.all(torch.norm((out_val-x),p=float("inf"),dim=(1,2)) > eps):
            #  break

        out_val.requires_grad_()
        pred,trans_feat = predict(out_val)
        #  out_pred = pred.data.max(1)[1]


        loss = loss_fn(pred,y.long(),trans_feat)
        predict.zero_grad()
        loss.backward()

        assert out_val.grad is not None

        # sign:BxDxN
        sign = alpha* out_val.grad.data.sign() 

        out_val = out_val.detach().clone()
        out_val += sign
        # TODO: batch clip or mask
        #  out_val = batch_clip_y(out_val,sign,fixed)



    return out_val,pred 


        


