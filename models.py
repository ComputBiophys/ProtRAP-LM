import torch
import numpy as np
import argparse,csv,sys
import os,requests
import torch.nn as nn
import torch.nn.functional as nnF


model_path=lambda x:'models/model_'+str(x)+'.pts'
github_url=lambda x:f"https://github.com/ComputBiophys/ProtRAP-LM/releases/download/Version1.0/model_{str(x)}.pts"

def download_file(url, output_path):
    try:
        response = requests.get(url)
        response.raise_for_status()  # 检查请求是否成功
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded file from {url} to {output_path}")
    except Exception as e:
        print(f"Error downloading file: {e}, You may manually download this one")

for i in range(10):
    if not os.path.exists(model_path(i)):
        print('Downloading model_'+str(i))
        download_file(github_url(i), model_path(i))


def fasta_load(fasta_dir):
    fp = open(fasta_dir, 'r')
    lines = fp.readlines()
    fp.close()
    sequence = ''
    for line in lines[1:]:
        sequence = sequence + line.split()[0]
    return sequence
def weight_MSE_loss(labels,logits,weights=1):
    l=(labels-logits)**2
    l=l*weights
    return torch.sum(l)
def focal_loss_softmax(labels,logits):
    y_pred=logits
    l=-labels*torch.log(y_pred+1e-8)*((1-y_pred)**2)
    return torch.sum(l)

class MultiScaleCNN(nn.Module):
    def __init__(self,input_dim=1280,output_dim=256):#,size=[3,7,11],padding=[1,3,5]):
        super().__init__()
        self.cnn1=nn.Conv1d(input_dim,output_dim,3,padding=1)
        self.cnn2=nn.Conv1d(input_dim,output_dim,5,padding=2)
        self.cnn3=nn.Conv1d(input_dim,output_dim,7,padding=3)
        self.cnn4=nn.Conv1d(input_dim,output_dim,9,padding=4)
    def forward(self,x):
        x=x.permute(0,2,1)
        x1=self.cnn1(x)
        x2=self.cnn2(x)
        x3=self.cnn3(x)
        x4=self.cnn4(x)
        x=torch.cat((x1,x2,x3,x4), -2)
        x=x.permute(0,2,1)
        return x
        
class ProtRAP_LM_Model(nn.Module):
    def __init__(self,input_dim=1280,n_hidden=256,num_layers=2,dropout=0.1):
        super().__init__()
        assert n_hidden%8==0

        self.keep_prob=1-dropout
        self.begin_linears=nn.Sequential(
            nn.Linear(input_dim,n_hidden*2),nn.ReLU(),nn.Dropout(self.keep_prob),
            nn.Linear(n_hidden*2,n_hidden),)
        self.cnn=MultiScaleCNN(input_dim=n_hidden,output_dim=int(n_hidden/4))
        encoder_layer=nn.TransformerEncoderLayer(d_model=n_hidden, nhead=4,activation='gelu',batch_first=True)
        self.encoder= nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
        self.pred=nn.Sequential(
            nn.Linear(n_hidden,int(n_hidden/2)),nn.ReLU(),nn.Dropout(self.keep_prob),nn.Linear(int(n_hidden/2),64),
            nn.Linear(64,3),nn.Sigmoid())
        return
    def forward(self,x):
        x=self.begin_linears(x)
        x=self.cnn(x)+x
        x=self.encoder(x)
        prediction=self.pred(x)
        return prediction

class ProtRAP_LM():

    def __init__(self,device_name='cpu'):
        device = torch.device(device_name)
        self.device=device

        esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        batch_converter = alphabet.get_batch_converter()
        esm_model=esm_model.eval().to(device)
        models=[]
        for i in range(10):
            model=torch.jit.load('./models/model_'+str(i)+'.pts').to(device).eval()
            models.append(model)
        self.models=models
        self.esm_model=esm_model
        self.batch_converter=batch_converter
        
    def predict(self,seq):
        data=[('prot',seq)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens=batch_tokens.to(self.device)
        preds=[]
        with torch.no_grad():
            results=self.esm_model(batch_tokens,repr_layers=[33])
            Repr= results["representations"][33]
            for model in self.models:
                pred=model(Repr).to(torch.device("cpu"))
                preds.append(np.array(pred[0,1:-1,:]))
        preds=np.array(preds)
        mean_pred=np.mean(preds,axis=0)
        std_pred=np.std(preds,axis=0)
        result=np.concatenate((mean_pred,std_pred),axis=-1)
        return result
