import os
import sys 
sys.path.append(os.getcwd()) # NOQA

import torch 
import datetime
from tqdm import tqdm
from src.dnn.utils import compute_eer, save_pickle

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def train(model, optimizer, criterion, data_loader, num_epochs, validation_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    train_loss=[]
    eer = []    
    print("Start training process")
    for epoch in range(num_epochs):
        model.train().to(device)
        running_loss = []
        print(f'EPOCH {epoch}:')
        for idx, data in enumerate(tqdm(data_loader)) :
            first_utterance, second_utterance, label = data
            
            optimizer.zero_grad()
            output1, output2 = model(first_utterance, second_utterance)
            loss = criterion(output1, output2, label)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0 :
                print(f"Batch {idx}: Loss: {loss.item()}")
        
        print(f"Epoch {epoch}: Loss average= {sum(train_loss) / len(train_loss)}")
        
        # Eval to check the eer score
        print("EVAL:")
        model.eval()
        epoch_output = []
        epoch_label = []

            # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vfirst_input, vsecond_input, vlabel = vdata
                voutput1, voutput2 = model(vfirst_input, vsecond_input)
                voutput = torch.nn.functional.cosine_similarity(voutput1, voutput2)
                epoch_output.append(voutput.item())
                epoch_label.append(int(vlabel))
            current_eer = compute_eer(epoch_label, epoch_output, positive_label= 1)
            print("CURRENT EER IS: ", current_eer)
            eer.append(current_eer)
        
    
                
        model_path = 'model_{}_epoch{}_eer= {}'.format(timestamp, epoch, current_eer)
        torch.save(model, f'/kaggle/working/{model_path}.pth')

    save_pickle(eer, filename= "eer.pk")
        
        