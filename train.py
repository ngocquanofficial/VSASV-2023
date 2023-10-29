import torch 
import torch.nn as nn
import numpy as np
from model import TripletLoss, Model
import datetime
from tqdm import tqdm

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
def get_accuracy(pred_arr,original_arr):
    pred_arr = pred_arr.detach().numpy()
    original_arr = original_arr.numpy()
    final_pred= []

    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0

    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count+=1
    return count/len(final_pred)*100

def train(model, optimizer, criterion, data_loader, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    train_loss=[]
    
    print("Start training process")
    for epoch in range(num_epochs):
        model.train().to(device)
        running_loss = []
        print(f'EPOCH {epoch}:')
        for idx, data in enumerate(tqdm(data_loader)) :
            target_sv_emb, second_sv_emb, second_antisf_emb, label = data
            
            optimizer.zero_grad()
            output = model(target_sv_emb, second_sv_emb, second_antisf_emb)
            loss = criterion(nn.Sigmoid(output), label)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0 :
                print(f"Batch {idx}: Loss: {loss.item()}")
        
        print(f"Epoch {epoch}: Loss average= {sum(train_loss) / len(train_loss)}")
    
        model_path = 'model_{}_at_epoch{}'.format(timestamp, epoch)
        torch.save(model, f'/kaggle/working/{model_path}.pth')

def train_with_triplet_loss(model, optimizer, criterion, data_loader, num_epochs) :
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    train_loss=[]
    
    print("Start training process")
    for epoch in range(num_epochs):
        model.train().to(device)
        running_loss = []
        print(f'EPOCH {epoch}:')
        for idx, data in enumerate(tqdm(data_loader)) :
            anchor_embs, positive_embs, negative_embs = data
            
            optimizer.zero_grad()
            anchor = model(*anchor_embs)
            positive = model(*positive_embs)
            negative = model(*negative_embs)
            loss = criterion(anchor, positive, negative)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0 :
                print(f"Batch {idx}: Loss: {loss.item()}")
        
        print(f"Epoch {epoch}: Loss average= {sum(train_loss) / len(train_loss)}")
    
        model_path = 'model_{}_at_epoch{}'.format(timestamp, epoch)
        torch.save(model, f'/kaggle/working/{model_path}.pth')


## Just for testing
# data_loader = [(torch.rand(192, 1), torch.rand(192, 1), torch.rand(160, 1)), (torch.rand(192, 1), torch.rand(192, 1), torch.rand(160, 1))]
            
# from model import Model
# model = Model()
# criterion = torch.nn.MSELoss()     
# test(model= model, criterion= criterion, data_loader= data_loader)      
        
    
