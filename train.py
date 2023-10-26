import torch 
import numpy as np
from model import TripletLoss, Model
import datetime
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
    train_loss=[]
    
    print("Start training process")
    for epoch in range(num_epochs):
        model.train().to(device)
        running_loss = []
        print(f'EPOCH {epoch}:')
        for i, data in enumerate(data_loader) :
            target_sv_emb, second_sv_emb, second_antisf_emb, label = data
            optimizer.zero_grad()
            output = model(target_sv_emb, second_sv_emb, second_antisf_emb)
            loss = criterion(output, label)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0 :
                print(f"Batch {i}: Loss: {loss.item()}")
        
        print(f"Epoch {epoch}: Loss average= {sum(train_loss) / len(train_loss)}")
    
        model_path = 'model_{}_at_epoch{}'.format(timestamp, epoch)


