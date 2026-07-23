from model import get_model, cheXNet
from chestxray_dataset import get_dataloaders
from exception import CustomException

import os
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from logger import setup_logger

logging = setup_logger()

DEBUG = True

def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr= 0.0001)

def get_criterion(pos_weights, device):
    return nn.BCEWithLogitsLoss(pos_weight=pos_weights).to(device)

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    
    model.train()
    loss = 0.0
    for batch_idx, (image, label) in enumerate(loader):
        try:
            if DEBUG and batch_idx >= 5:
                break
            
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)
            outputs = model(image)
            batch_loss = criterion(outputs, label)
            batch_loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logging.info(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {batch_loss.item():.4f}")
                # i move this line up
            loss += batch_loss.item()
            
        except CustomException as e:
            logging.info("exception",e)
            continue
        
    loss /= len(loader)
    return loss
        
def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    loss = 0.0
    with torch.no_grad():
        for batch_idx, (image, label) in enumerate(loader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            batch_loss = criterion(output, label).item()
            loss += batch_loss
            if batch_idx % 100 == 0:
                logging.info(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {batch_loss.item():.4f}")

    loss /= len(loader)
    return loss

def save_checkpoint(model, epoch, val_loss, path):
    checkpoint = {
        'model_state' : model.state_dict(),
        'epoch' : epoch,
        'val_loss' : val_loss,
        }
    save_path = f"{path}/checkpoint_epoch{epoch}.tar"
    torch.save(checkpoint, save_path)
    print(f"  Checkpoint saved {save_path}")

def main():
    if DEBUG :
        logging.info("Started training in Debugging mode...")
        batch_size = 4
        num_workers = 0
    else:
        logging.info("Started training...")
        batch_size = 32
        num_workers = 4
    train_loader, val_loader, _, pos_weights = get_dataloaders(Path("../dataset/"),batch_size=batch_size,num_workers=num_workers)
    logging.info("Loader(s) and positional weights are loaded successfully")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = get_model(num_classes=14, device=device)
    optimizer = get_optimizer(model=model)
    criterion = get_criterion(pos_weights=pos_weights, device=device)
    if not optimizer or not criterion or not model:
        logging.error("model did not loaded error either in model, optimizer, criterion")

    epochs = 10
    
    best_val_loss = float("inf")
    logging.info(f"Using {epochs} epochs..")
    start_time = datetime.now()
    for epoch in range(epochs):
        try:
            train_loss = train_one_epoch(model=model,
                                        loader=train_loader, 
                                        criterion=criterion, 
                                        optimizer= optimizer, device= device,
                                        epoch = epoch)
        
            val_loss = evaluate(model=model, 
                                    loader=val_loader, 
                                    criterion=criterion, device=device)
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, 
                            mode='min',        # reduce when val_loss stops going down
                            patience=2,        # wait 2 epochs before reducing
                            factor=0.1         # multiply lr by 0.1
                            )

            scheduler.step(val_loss)
            
            logging.info(f"Epoch {epoch+1}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

            # Only save if this is the best model so far
            patience  = 5
            epochs_no_improve = 0

            if val_loss < best_val_loss:
                epochs_no_improve = 0
                best_val_loss = val_loss
                check_point_path = Path.joinpath(Path(__file__).parent,"checkpoint")
                os.makedirs(check_point_path, exist_ok=True)
                save_checkpoint(model, epoch+1, val_loss, check_point_path)
                logging.info(f"  New best model (val_loss: {val_loss:.4f})")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logging.info("Early stopping triggered")
                    break
                
        except CustomException as e:
            logging.info("Exception",e)
            continue
    end_time = datetime.now()
    total_time_run = end_time - start_time
    logging.info(f"Training completed in {total_time_run} !!!")
    return model

if __name__ == "__main__":
    main()