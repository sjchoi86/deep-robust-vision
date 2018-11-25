import os,glob,cv2,warnings,time,sys,itertools
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
from tensorflow.examples.tutorials.mnist import input_data

class nzr(object):
    def __init__(self,_rawdata,_eps=1e-8):
        self.rawdata = _rawdata
        self.eps     = _eps
        self.mu      = np.mean(self.rawdata,axis=0)
        self.std     = np.std(self.rawdata,axis=0)
        """ GET NORMALIZED VAL """
        self.nzd_data = self.get_nzdval(self.rawdata)
        """ GET  ORIGIANL VAL """
        self.org_data = self.get_orgval(self.nzd_data)
        """ CHECK ERROR """
        self.maxerr = np.max(self.rawdata-self.org_data)
    def get_nzdval(self,_data):
        _n = _data.shape[0]
        _nzddata = (_data - np.tile(self.mu,(_n,1))) / np.tile(self.std+self.eps,(_n,1))
        return _nzddata
    def get_orgval(self,_data):
        _n = _data.shape[0]
        _orgdata = _data*np.tile(self.std+self.eps,(_n,1))+np.tile(self.mu,(_n,1))
        return _orgdata

def get_dataset(_loadpath='data/',_rszshape=(28,28,1),_imgext='png',_VERBOSE=True):
    flist  = sorted(os.listdir(_loadpath))
    nclass = len(flist)
    
    # 1. Compute the total number of images
    n_total  = 0
    for fidx,fn in enumerate(flist): # For all folders
        plidst = sorted(glob.glob(_loadpath+fn+'/*.'+_imgext))
        if _VERBOSE:
            print ("[%d/%d] [%04d] images" %(fidx,nclass,len(plidst)))
        n_total = n_total + len(plidst)
    
    # 2.  Load Data 
    if _VERBOSE: print ("Start loading total [%d] images." % (n_total))
    X = np.zeros((n_total,_rszshape[0]*_rszshape[1]*_rszshape[2]))
    Y = np.zeros((n_total,nclass))
    imgcnt = 0
    for fidx,fn in enumerate(flist): # For all folders
        plidst = sorted(glob.glob(_loadpath+fn+'/*.'+_imgext))
        for pn in plidst: # For all images per folder   
            if _rszshape[2] == 1: # If the last channel is 1 then, make grayscale. 
                img_raw = cv2.imread(pn, cv2.IMREAD_GRAYSCALE)
            else: 
                img_raw = cv2.imread(pn, cv2.IMREAD_COLOR)
                img_raw = cv2.cvtColor(img_raw,cv2.COLOR_BGR2RGB)
            img_rsz = cv2.resize(img_raw,_rszshape[:2])
            img_vec = img_rsz.reshape((1,-1))/255.
            """ Concatenate input and output to X and Y """
            X[imgcnt:imgcnt+1,:] = img_vec
            Y[imgcnt:imgcnt+1,:] = np.eye(nclass, nclass)[fidx:fidx+1,:]
            imgcnt = imgcnt + 1
    if _VERBOSE:
        print ('Done.')
        
    # 3. Ramdom Shuffle with Fixed Random Seed 
    np.random.seed(seed=0) # Fix seed 
    randidx = np.random.randint(imgcnt,size=imgcnt)
    X = X[randidx,:]
    Y = Y[randidx,:]
    return X, Y, imgcnt

def load_mnist():
    mnist = input_data.read_data_sets('../data/', one_hot=True)
    trainimg,trainlabel = mnist.train.images,mnist.train.labels
    testimg,testlabel = mnist.test.images,mnist.test.labels
    valimg,vallabel = mnist.validation.images,mnist.validation.labels
    return trainimg,trainlabel,testimg,testlabel,valimg,vallabel

def load_mnist_with_noise(_errType='rs',_outlierRatio=0.00,_seed=0):
    # Load MNIST 
    trainimg,trainlabel,testimg,testlabel,valimg,vallabel = load_mnist()
    if _outlierRatio == 0:
        return trainimg,trainlabel,testimg,testlabel,valimg,vallabel
    
    # Add outliers 
    if _errType == 'rs': # Random Shuffle
        np.random.seed(seed=_seed); 
        outlierRatio = _outlierRatio
        nOutlier = (int)(outlierRatio*trainimg.shape[0])
        oIdx = np.random.permutation(trainimg.shape[0])[:nOutlier]
        trainlabel[oIdx,:] = np.eye(10)[np.random.choice(10,nOutlier)]
    elif _errType == 'rp':# Random Perturbation (from Reed)
        _outlierRatio /= 2.0 # For random perturbation, half the error ratio! 
        perm = np.array([7, 9, 0, 4, 2, 1, 3, 5, 6, 8])
        X_train,y_train = trainimg,np.argmax(trainlabel,axis=1)
        noise = perm[y_train]
        from sklearn.model_selection import StratifiedShuffleSplit
        _, noise_idx = next(iter(StratifiedShuffleSplit(n_splits=1,
                            test_size=_outlierRatio,
                            random_state=_seed).split(X_train,y_train)))
        y_train_noise = y_train.copy() 
        y_train_noise[noise_idx] = noise[noise_idx]
        trainlabel = np.eye(10)[y_train_noise]
    elif _errType == 'None':
        DO_NOTHING = True
    else:
        print ("Unknown error type: [%s]."%(_errType))
    return trainimg,trainlabel,testimg,testlabel,valimg,vallabel

def plot_rand_imglabels_inarow(_X,_Y,_labels,_rszshape,_nPlot):
    f,axarr = plt.subplots(1,_nPlot,figsize=(18,8))
    for idx,imgidx in enumerate(np.random.randint(_X.shape[0],size=5)):
        currimg=np.reshape(_X[imgidx,:],_rszshape).squeeze()
        currlabel=_labels[np.argmax(_Y[imgidx,:])]
        if _rszshape[2]==1: axarr[idx].imshow(currimg,cmap=plt.get_cmap('gray'))
        else: axarr[idx].imshow(currimg)
        axarr[idx].set_title('[%d] %s'%(imgidx,currlabel),fontsize=15)
    plt.show()
    
def plot_imglabels_inarow(_X,_Y,_labels,_rszshape):
    nImg = _X.shape[0]
    f,axarr = plt.subplots(1,nImg,figsize=(18,8))
    for idx in range(nImg):
        currimg=np.reshape(_X[idx,:],_rszshape).squeeze()
        currlabel=_labels[np.argmax(_Y[idx,:])]
        if _rszshape[2]==1: axarr[idx].imshow(currimg,cmap=plt.get_cmap('gray'))
        else: axarr[idx].imshow(currimg)
        axarr[idx].set_title('[%d] %s'%(idx,currlabel),fontsize=15)
    plt.show()
    
def augment_img(_imgVec,_imgSize):
    # Reshape to image
    imgVecAug = np.copy(_imgVec)
    n = _imgVec.shape[0]
    imgs = np.reshape(_imgVec,[n]+_imgSize)
    for i in range(n):
        cImg = imgs[i,:,:,:] # Current img
        # Rotate
        angle = np.random.randint(-20,20,1)
        cImg = ndimage.rotate(cImg,angle,reshape=False
                                 ,mode='reflect',prefilter=True,order=1)
        # Flip
        if np.random.rand()>0.5: cImg = np.fliplr(cImg)
        # Shift
        shift = np.random.randint(-3,3,3);shift[2]=0
        cImg = ndimage.shift(cImg,shift,mode='reflect')
        # Append
        imgVecAug[i,:] = np.reshape(cImg,[1,-1])
    imgVecAug = np.clip(imgVecAug,a_min=0.0,a_max=1.0)
    return imgVecAug

def gpusession(): 
    config = tf.ConfigProto(); 
    config.gpu_options.allow_growth=True
    # config.log_device_placement=False
    sess = tf.Session(config=config)
    return sess

def create_gradient_clipping(loss,optm,vars,clipVal=1.0):
    grads, vars = zip(*optm.compute_gradients(loss, var_list=vars))
    grads = [None if grad is None else tf.clip_by_value(grad,-clipVal,clipVal) for grad in grads]
    op = optm.apply_gradients(zip(grads, vars))
    train_op = tf.tuple([loss], control_inputs=[op])
    return train_op[0]

def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        print (_chars)


def extract_percent(_tokens,_key):
    _selItem = [x for x in _tokens if (_key in x) & ('%' in x)][0]
    _selItem = _selItem.replace(_key,'')
    _selItem = _selItem.replace(':','')
    _selItem = _selItem.replace('%','')
    return (float)(_selItem) 
def plot_cifar10_accuracy(_Accrs,_txtList,_title='Accuracy'):
    plt.figure(figsize=(8,5))
    _cmap = plt.get_cmap('gist_rainbow')
    _nConfig = _Accrs.shape[0]
    _colors = [_cmap(i) for i in np.linspace(0,1,_nConfig)]
    _max_cEpoch = 0
    for i in range(_nConfig): # For different configurations
        _cAccrs = _Accrs[i,:]
        _cEpoch = np.where(_cAccrs==0)[0][0]
        if _cEpoch>_max_cEpoch: _max_cEpoch=_cEpoch
        _cAccrs = _cAccrs[:_cEpoch] # Trim non-zero
        _fName = _txtList[i] 
        _fNameRfn = _fName.replace('../res/res_cifar10_','') # Remove header 
        _fNameRfn = _fNameRfn.replace('.txt','') # Remove .txt 
        if 'mcdn' in _fNameRfn: _ls = '-' # Solid line for MCDN
        else: _ls = '--' # Dotted line for CNN 
        plt.plot(_cAccrs,label=_fNameRfn,color=_colors[i],lw=2,ls=_ls,marker='')
    plt.xlim([0,_max_cEpoch])
    plt.ylim([0,100])
    plt.grid(b=True)
    plt.title(_title,fontsize=20);
    #plt.legend(fontsize=12,loc='lower left')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=2)
    plt.xlabel('Epoch',fontsize=15);plt.ylabel('Accuracy [%]',fontsize=15)
    plt.show()
# Parse accuracies
def parse_accuracies(_txtList):
    nConfig = len(_txtList) # Number of configurations
    trainAccrs = np.zeros(shape=(nConfig,(int)(1e3)))
    testAccrs = np.zeros(shape=(nConfig,(int)(1e3)))
    valAccrs = np.zeros(shape=(nConfig,(int)(1e3)))
    for fIdx,fName in enumerate(_txtList):
        nEpoch = sum(1 for line in open(fName)) # Number of epochs
        for lIdx,eachLine in enumerate(open(fName,'r')): # For each line
            if lIdx==0: continue
            trainAccr = extract_percent(eachLine.split(' '),'train')
            testAccr = extract_percent(eachLine.split(' '),'test')
            valAccr = extract_percent(eachLine.split(' '),'val')
            trainAccrs[fIdx,lIdx-1] = trainAccr
            testAccrs[fIdx,lIdx-1] = testAccr
            valAccrs[fIdx,lIdx-1] = valAccr
            fNameRfn = fName.replace('../res/res_cifar10_','')
            fNameRfn = fNameRfn.replace('.txt','')  
    return trainAccrs,testAccrs,valAccrs

class grid_maker(object): # For multi-GPU testing
    def __init__(self,*_arg):
        self.arg = _arg
        self.nArg = len(self.arg) # Number of total lists
        _product = itertools.product(*self.arg); _nIter = 0
        for x in _product: _nIter += 1
        self.nIter = _nIter
        self.paramList = ['']*self.nIter
        self.idxList = ['']*self.nIter
        _product = itertools.product(*self.arg);
        for idx,x in enumerate(_product):
            self.paramList[idx] = x
def get_properIdx(_processID,_maxProcessID,_nTask): # For multi-GPU testing
    ret = []
    if _processID > _nTask: return ret
    if _processID > _maxProcessID: return ret
    m = (_nTask-_processID-1) // _maxProcessID
    for i in range(m+1):
        ret.append(i*_maxProcessID+_processID)
    return ret

def mixup(data, targets, alpha):
    n = data.shape[0]
    indices = np.random.permutation(n)
    data2 = data[indices,:]
    targets2 = targets[indices,:]
    lam = np.random.beta(alpha,alpha,size=(n,1)) # Sample from beta. 
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)
    return data, targets







