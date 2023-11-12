import os
import sys 
sys.path.append(os.getcwd()) # NOQA

import torch 
import torch.nn as nn
import numpy as np
from src.fusion.LCNN.model.lcnn import LCNN
import datetime
from tqdm import tqdm
from src.naive_dnn.utils import compute_eer, save_pickle

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def train(model, optimizer, criterion, data_loader, num_epochs, validation_loader, model_type= "stft"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    train_loss=[]
    eer = []    
    print("Start training process")
    min_loss = 100
    for epoch in range(num_epochs):
        model.train().to(device)
        running_loss = []
        print(f'EPOCH {epoch}:')
        for idx, data in enumerate(tqdm(data_loader)) :
            wave, label = data
            
            optimizer.zero_grad()
            last_hidden, output = model(wave)
            loss = criterion(output, label)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
            if idx % 100 == 0 :
                print(f"Batch {idx}: Loss: {loss.item()}")
        
        print(f"Epoch {epoch}: Loss average= {sum(train_loss) / len(train_loss)}")
        
        # Eval to check the eer score and loss
        print("EVAL:")
        model.eval()
        epoch_output = []
        epoch_label = []
        eval_loss = 0.0
        num_sample = 0
        avg_eval_loss = 0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vwave, vlabel = vdata
                emb, voutput = model(vwave)

                epoch_output.append(voutput.argmax().item())
                epoch_label.append(int(vlabel))

                # Calculate loss on the evaluation set
                eval_loss += criterion(voutput, vlabel).item()
                num_sample += 1

            current_eer = compute_eer(epoch_label, epoch_output, positive_label=1)
            print(f"CURRENT EER IS: {current_eer} at epoch {epoch}")
            eer.append(current_eer)

            # Calculate average loss on the evaluation set
            avg_eval_loss = eval_loss / num_sample
            print(f"AVERAGE EVAL LOSS: {avg_eval_loss} at epoch {epoch}")

        if avg_eval_loss < min_loss :
            min_loss = avg_eval_loss
            model_path = 'model_{}_epoch{}'.format(model_type, epoch)
            torch.save(model, f'/kaggle/working/{model_path}.pth')
            print(f"Save model at epoch {epoch}")

    save_pickle(eer, filename= f"{model_type}_eer.pk")