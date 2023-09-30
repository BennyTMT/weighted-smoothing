from multiprocessing import Semaphore
from .misc import *
from .logger import *
from .visualize import *
from .eval import *
import numpy as np
import os, sys , random , torch , math  
import torch.nn as nn
initEp = 2 ;  
sys.path.append(os.path.join(os.path.dirname(__file__), "progress"))
use_cuda = True 
m_soft    = nn.Softmax(dim=1)
criterion = nn.NLLLoss() 

def m_entr_comp(probs, true_labels, numClass):
    mentr = []
    def log_value(probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
    def softmax(datas):
        sofx =[]
        for i in range(len(datas)):
            x   = datas[i]
            f_x = np.exp(x) / (np.sum(np.exp(x))+np.exp(-20) )
            sofx.append(f_x)
        return np.array(sofx).reshape(-1,numClass)

    trd = 20
    if np.max(probs) > trd: probs = np.clip(probs,-100000,20 )
    probs = softmax(probs)
    
    #计算mentr 
    for i in range(len(probs)):
        prob      = probs[i]
        label     = int(true_labels[i])
        reverProb = 1- prob
        logProb   = log_value(prob) ; log_reve_prob = log_value(reverProb)
        first  = reverProb * logProb; second = prob*log_reve_prob
        second[label] = first[label]
        mentr.append(np.sum(second))
    return np.array(mentr)
    
def updateWM_( LoadData , model,batch_size , numclass=-1, epoch = -100 ):
    model.eval()
    batchs = len(LoadData.inputs) // batch_size
    memLabs= [] ; members =[] ;
    with torch.no_grad(): 
        for ind in range(batchs):
            inputs  = torch.autograd.Variable(torch.from_numpy(  LoadData.inputs[ind*batch_size :(ind +1) * batch_size]))
            inputs= inputs.float().cuda()
            # compute output 
            attVector , _ = model(inputs)
            members.append(attVector.cpu().detach().numpy()) 

        members = np.array(members).reshape((-1,numclass)) ;memLabs = LoadData.targets
        
        assert  len(members) == len(memLabs)
        mentr   = m_entr_comp(members , LoadData.targets ,numclass )
        # 将mentr 分配给每个数据
        LoadData.weightMatrix = [np.zeros((0,1))] * numclass
        for idx in range(len(memLabs)):
            lab  = int(memLabs[idx]) 
            LoadData.weightMatrix[lab] = np.append(LoadData.weightMatrix[lab], 
                                    mentr[idx].reshape((1,1)))

        for i in range(numclass):
            wm = LoadData.weightMatrix[i] + math.exp(-16)
            # (X,1)
            # wm = logMentr(wm) ; 
            # wm = (wm - np.min(wm)) / (np.max(wm) - np.min(wm) + np.exp(-36)); np.clip(wm , 0 , 1 ) # 1 严格[0,1]
            wm = (wm - np.mean(wm)) / (np.std(wm) +  np.exp(-20)) ;
            LoadData.weightMatrix[i] = wm
        # 逆向collection 
        locations =[0] * numclass  ;LoadData.smuWM =[]
        for idx in range(len(memLabs)):
            lab  = int(memLabs[idx]) 
            LoadData.smuWM.append( LoadData.weightMatrix[lab][locations[lab]])
            locations[lab] +=1 

        # check 
        for i in range(numclass):
            assert (locations[i]== len(LoadData.weightMatrix[i]) ) 

def calculate_cross_entropy(vec1, vec2):
    # One-hot encode vec2
    n_labels = vec1.shape[1]
    one_hot = np.zeros((vec2.size, n_labels))
    one_hot[np.arange(vec2.size), vec2] = 1
    # Compute the cross entropy
    return -np.sum(one_hot * np.log( np.clip(vec1, 1e-10, 1 - 1e-10)) , axis=1 ) 

def updateWM_crossentropy( LoadData , model,batch_size , numclass=-1, epoch = -100 ):
    global fixed_mean , fixed_std 
    model.eval()
    batchs = len(LoadData.inputs) // batch_size
    memLabs= [] ; members =[] ;
    with torch.no_grad(): 
        for ind in range(batchs):
            inputs  = torch.autograd.Variable(torch.from_numpy(  LoadData.inputs[ind*batch_size :(ind +1) * batch_size]))
            inputs= inputs.float().cuda()
            # compute output 
            attVector , _ = model(inputs)
            members.append(attVector.cpu().detach().numpy()) 

        members = np.array(members).reshape((-1,numclass)) ;memLabs = LoadData.targets
        
        assert len(members) == len(memLabs) 
        mentr = calculate_cross_entropy(members, LoadData.targets )

        # 将mentr 分配给每个数据
        LoadData.weightMatrix = [np.zeros((0,1))] * numclass
        for idx in range(len(memLabs)):
            lab  = int(memLabs[idx]) 
            LoadData.weightMatrix[lab] = np.append(LoadData.weightMatrix[lab], 
                                    mentr[idx].reshape((1,1)))

        for i in range(numclass):
            wm = LoadData.weightMatrix[i] + math.exp(-16)
            wm = (wm - np.mean(wm)) / (np.std(wm) +  np.exp(-20)) ; #
            LoadData.weightMatrix[i] = wm

        locations =[0] * numclass  ;LoadData.smuWM =[]
        for idx in range(len(memLabs)):
            lab  = int(memLabs[idx]) 
            LoadData.smuWM.append( LoadData.weightMatrix[lab][locations[lab]])
            locations[lab] +=1 

        # check 
        for i in range(numclass):
            assert locations[i] == len(LoadData.weightMatrix[i])

def trainTdp_bound_predic(LoadData, model, criterion, optimizer, epoch ,batch_size ,randStd , senBound = -1, 
                   numclass=-1 ):
    if senBound == -1 or numclass == -1 : print('senBound == -1 or numclass == -1')  ; exit()
    def projection(predic_vector):
        absPred_  = torch.abs(predic_vector) 
        absPred_ /= torch.sum(absPred_ , axis = 1 , keepdims=True )
        absPred_  = torch.clip(absPred_, math.exp(-36) , 1 )
        return absPred_ 
        
    def getHighLow(vector  ):
        sorts  = np.sort(vector, axis=1)
        return   sorts[:,-1] - sorts[:,-2] # 最大 - 次大 

    # switch to train mode
    model.train()
    losses = AverageMeter() ; top1 = AverageMeter() ; top5 = AverageMeter()
    nInputs = LoadData.inputs ; nTargets = LoadData.targets ; weihgtMatrix =  LoadData.smuWM
    batchs = len(nInputs) // batch_size
    # nInputs , nTargets , weihgtMatrix = randomSample(nInputs , nTargets , weihgtMatrix)

    if len(LoadData.smuWM) ==0  and epoch == 50 :  print("epoch :" , epoch , " no noise added ! ")
    
    for ind in range(batchs):
        inputs  = torch.autograd.Variable(torch.from_numpy(nInputs[ ind*batch_size :(ind +1) * batch_size]))
        tarB    = nTargets[ind*batch_size :(ind +1) * batch_size] ; targets = torch.autograd.Variable(torch.from_numpy(tarB))
        inputs, targets = inputs.float().cuda(), targets.cuda(non_blocking=True)
        # compute output
        logit_ , _   = model(inputs)
        # print(logit_.shape , e.shape  )
        predic_      = m_soft(logit_)
        # Noise Generator 
        decys    = np.zeros((batch_size,numclass)) 
        if len(weihgtMatrix) > 0:
            # Finished Initialization 
            pred_Numpy   = predic_.cpu().detach().numpy().astype(np.float32)
            wm       = weihgtMatrix[ ind*batch_size :(ind +1) * batch_size]
            upNoises = getHighLow(pred_Numpy  )
            for m in range(batch_size):
                # Sigma Gaussian
                decys[m]= np.random.normal(0,randStd, ( 1,numclass ))  * float(upNoises[m])   *( 1- wm[m] )

        decys   = torch.autograd.Variable(torch.from_numpy(decys)).float().cuda()
        predic_ = predic_ +  decys
        predic_ = projection(predic_)
        
        predic_ = torch.log(predic_ )
        loss    = criterion(predic_, targets)  # printNoise()
        # measure accuracy and record loss
        prec1, prec5 = accuracy(logit_.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
        optimizer.step()
        
    if  randStd > 1e-12 and epoch > initEp :   updateWM_( LoadData , model,batch_size , numclass=numclass , epoch=epoch )
    
    return losses.avg, top1.avg   

def testTdp(LoadData, model, criterion, epoch, use_cuda ,batch_size =-1 ):
    global best_acc
    criterion = nn.CrossEntropyLoss()
    if batch_size == -1 : print("batch_size == -1 " ) ; exit()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    batchs = len(LoadData.testInp) // batch_size
    
    for ind in range(batchs):
        
        inputs  = torch.autograd.Variable(torch.from_numpy(LoadData.testInp[ind*batch_size :(ind +1) * batch_size]))
        targets = torch.autograd.Variable(torch.from_numpy(LoadData.testTag[ind*batch_size :(ind +1) * batch_size]))
        if use_cuda:
            inputs, targets = inputs.float().cuda(), targets.cuda(non_blocking=True)

        # compute output
        outputs ,_ = model(inputs)
        loss       = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))

        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    return (losses.avg, top1.avg)

def saveVitData(LoadData, model   , batch_size, senBound=-1
                     , sp='a' , randStd=-1  , embSize = -1 ,  predSize= -1 , ABgroup=0 ):

    if sp=='a' or  embSize == -1 or \
            predSize== -1 or randStd==-1 or senBound ==-1 : print('err dir');  exit()

    model.eval()
    batchs = len(LoadData.inputs) // batch_size
    ofNum  =  str(randStd) + '_' + str(senBound) +'_' + str(ABgroup)
    memEmbs= [] ; memLabs= [] ; members =[] 
    for ind in range(batchs):
        inputs  = torch.autograd.Variable(torch.from_numpy(LoadData.inputs[ind*batch_size :(ind +1) * batch_size]))
        inputs= inputs.float().cuda()
        # compute output
        attVector , emb = model(inputs)

        members.append(attVector.cpu().detach().numpy())
        memLabs.append(LoadData.targets[ind*batch_size :(ind +1) * batch_size])
        memEmbs.append(emb.cpu().detach().numpy())
        
    members   = np.array(members).reshape(-1,predSize)
    memEmbs   = np.array(memEmbs).reshape(-1,embSize)
    memLabs   = np.array(memLabs).reshape(-1,)

    np.save( sp+'/member'    + str(ofNum) + '.npy'  , members)
    np.save( sp+'/memberEmb' + str(ofNum) + '.npy'  , memEmbs)
    np.save( sp+'/memberLab' + str(ofNum) + '.npy'  , memLabs)
    
    
    nonMembers= [] ; nonMemberE= [] ; nLabs  = []
    batchs = len(LoadData.testInp) // batch_size
    for ind in range(batchs):
        inputs  = torch.autograd.Variable(torch.from_numpy(LoadData.testInp[ind*batch_size :(ind +1) * batch_size]))
        inputs= inputs.float().cuda()
        # compute output
        attVector , emb = model(inputs)
        nonMembers.append(attVector.cpu().detach().numpy())
        nonMemberE.append(emb.cpu().detach().numpy())
        nLabs.append(LoadData.testTag[ind*batch_size :(ind +1) * batch_size])

    nonMembers = np.array(nonMembers).reshape(-1,predSize)
    nonMemberE = np.array(nonMemberE).reshape(-1,embSize)
    nonLabs    = np.array(nLabs).reshape(-1,)
    np.save( sp+'/nonMember' +    str(ofNum) + '.npy', nonMembers)
    np.save( sp+'/nonMemberEmb' + str(ofNum) + '.npy', nonMemberE)
    np.save( sp+'/nonMemberLab' + str(ofNum) + '.npy', nonLabs)

    print(ofNum)
    z = np.argmax(members, axis=1)==memLabs
    print(z , np.sum(z) ,  np.sum(z)/len(z))
    z = np.argmax(nonMembers, axis=1)==nonLabs
    print(z , np.sum(z) ,  np.sum(z)/len(z) )
    print()

    print("Get Prediction ... ")
    print('     Member:'    ,  members.shape , ' memLab:', memLabs.shape  , memEmbs.shape  )
    print('     NonMember:' , nonMembers.shape ,' nonMemLab:' ,nonLabs.shape, nonMemberE.shape )

    if len(nonMembers) > len(members) : 
        randomize = np.arange(len(nonMembers)); np.random.shuffle(randomize)
        nonMembers = nonMembers[randomize][:len(members)]  ; nonLabs  = nonLabs[randomize][:len(members)] ; 
    else:
        randomize = np.arange(len(members)); np.random.shuffle(randomize)
        members = members[randomize][:len(nonMembers)]  ; memLabs  = memLabs[randomize][:len(nonMembers)] ; 
    print("*" * 100) 

    attMode = attackBenchMark(members , memLabs  , nonMembers , nonLabs , num_classes=predSize )
    att= [ ]
    for method in ['confidence' , 'entropy' ,  'mentr'  ]:
        mem_inf_acc = attMode.attackBenchMark(method)
        att.append(mem_inf_acc)
    dataLen      = min(len(members), len(nonMembers)) ; tar = ofNum ; dPath = sp 
    print(dataLen)
    mem_inf_acc  = attackAcc(tar , dPath , predSize,dataLen)
    att.append(mem_inf_acc)

    return att 
