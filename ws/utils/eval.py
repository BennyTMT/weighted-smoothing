from __future__ import print_function, absolute_import
DB = False 

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    
    if DB : print('9 - accuracy : ' ,output.shape,  target.shape)

    _, pred = output.topk(maxk, 1, True, True)

    if DB : print('max : ' ,  maxk , 'pred : ', pred.shape)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:

        if DB : 
            print( 'correct : ', correct.shape)
            print(' correct[:k]' , correct[:k].shape,  k )

        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res