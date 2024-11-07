import torch
import torch.nn as nn
import numpy as np

def np2tensor(v):
    return torch.from_numpy(v)

def np2param(v, grad=True):
    return nn.Parameter(np2tensor(v), requires_grad=grad)

def tensor2np(v):
    return v.numpy()

def get_device():
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device

class RRRGD():
    def __init__(self, train_data, ncomp, l2=0.):
        self.l2=l2
        self.eids = list(train_data.keys())

        np.random.seed(0)
        self.N = 0; self.model = {}
        for eid in train_data:
            _X = train_data[eid]['X'][0] # (K,T,ncoef), the last coef is the bias term
            _y = train_data[eid]['y'][0] # (K,T,N)
            K, T, ncoef = _X.shape
            K, T, N = _y.shape
            U = np.random.normal(size=(N, ncoef-1, ncomp))/np.sqrt(T*ncomp)
            V = np.random.normal(size=(ncomp, T))/np.sqrt(T*ncomp)
            b = np.expand_dims(_y.mean(0).T, 1)
            b = np.ascontiguousarray(b)
            self.model[f"{eid}_U"]=np2param(U)
            self.model[f"{eid}_b"]=np2param(b)
            self.N += N
        self.model['V'] = np2param(V) # V shared across sessions
        self.n_comp, self.T = self.model['V'].shape
        self.model = nn.ParameterDict(self.model)
        # U: model[eid+"_U"], (N, ncoef, ncomp)
        # V: model['V'], (ncomp, T)
        # b: model[eid+"_b"], (N, 1, T)

    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model.to(device)

    def state_dict(self):
        checkpoint = {"model": {k: v.cpu() for k, v in self.model.state_dict().items()},
                      "l2": self.l2,
                      "eids": self.eids,
                      "N": self.N,
                      "T": self.T,
                      "n_comp": self.n_comp,}
        return checkpoint
    
    def load_state_dict(self, f):
        self.model.load_state_dict(f)

    """
    * input has to be tensor
    """
    def compute_beta_m(self, U, V, b, withbias=True, tonp=False):
        if tonp == True:
            U = np2tensor(U)
            V = np2tensor(V)
        beta = U @ V
        if withbias:
            if tonp == True:
                b = np2tensor(b)
            beta = torch.cat((beta, b), 1) # (N, ncoef, T)
        else:
            # pass # previously used
            # place-holder
            b = torch.zeros((U.shape[0], 1, V.shape[1])).to(beta.device)
            beta = torch.cat((beta, b), 1) # (N, ncoef, T)
        if tonp == True:
            beta = tensor2np(beta)
        
        return beta
        

    def compute_beta(self, eid, withbias=True):
        return self.compute_beta_m(self.model[f"{eid}_U"],
                                          self.model['V'],
                                          self.model[f"{eid}_b"],
                                          withbias=withbias)

    def predict(self, beta, X, tonp=False):
        """
        :beta: (N, ncoef, T)
        :x: (K, T, ncoef)
        """
        if tonp == True:
            X = np2tensor(X)
            beta = np2tensor(beta)
        y_pred = torch.einsum("ktc,nct->ktn", X, beta)
        if tonp == True:
            y_pred = tensor2np(y_pred)
        return y_pred

    """
    - input nparray 
    - output tensor
    """
    def predict_y(self, data, eid, k):
        # X: train_data[eid]['X'][0], (K, T, ncoef+1)
        # y: train_data[eid]['y'][0], (K, T, N)
        beta = self.compute_beta(eid, withbias=False)  # new: no bias
        X = np2tensor(data[eid]['X'][k]).to(beta.device)
        y = np2tensor(data[eid]['y'][k]).to(beta.device)
        # ypred = predict_torch(beta, X)
        ypred = self.predict(beta, X)
        return X, y, ypred

    """
    - input nparray 
    - output tensor
    """
    def predict_y_fr(self, data, eid, k):
        X, y, ypred = self.predict_y(data, eid, k)
        mean_y = np2tensor(data[eid]['setup']['mean_y_TN']).to(y.device)
        std_y = np2tensor(data[eid]['setup']['std_y_TN']).to(y.device)
        y = y * std_y + mean_y
        ypred = ypred * std_y + mean_y
        return X, y, ypred
    
    """
    - input and output nparray by default
    """
    def compute_MSE_RRRGD(self, data, k):
        mses_all = {}
        for eid in data:
            _, y, ypred = self.predict_y(data, eid, k)
            mses_all[eid] = torch.sum((ypred - y) ** 2, axis=(0, 1))
        return mses_all

    def regression_loss(self):
        return {eid: self.l2*torch.sum(self.compute_beta(eid, withbias=False)**2) for eid in self.eids}