import torch
import numpy as np
import argparse,csv,sys
import os

def fasta_load(fasta_dir):
    fp = open(fasta_dir, 'r')
    lines = fp.readlines()
    fp.close()
    sequence = ''
    for line in lines[1:]:
        sequence = sequence + line.split()[0]
    return sequence
    
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
