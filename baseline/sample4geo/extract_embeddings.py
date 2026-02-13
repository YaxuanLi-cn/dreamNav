import os
import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader

from sample4geo.dataset.university import U1652DatasetEval, get_transforms
from sample4geo.evaluate.university import evaluate
from sample4geo.model import TimmModel
import cv2
import pickle
from tqdm import tqdm

data_dir = '../../pairUAV/tours/'
output_dir = './embedding/'

@dataclass
class Configuration:

    # Model
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    
    # Override model image size
    img_size: int = 384
    
    # Evaluation
    batch_size: int = 128
    verbose: bool = True
    gpu_ids: tuple = (0,)
    normalize_features: bool = True
    eval_gallery_n: int = -1             # -1 for all or int
    
    
    # Checkpoint to start from
    checkpoint_start = 'pretrained/university/convnext_base.fb_in22k_ft_in1k_384/weights_e1_0.9515.pth'
  
    # set num_workers to 0 if on Windows
    num_workers: int = 0 if os.name == 'nt' else 4 
    
    # train on GPU if available
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu' 
    

#-----------------------------------------------------------------------------#
# Config                                                                      #
#-----------------------------------------------------------------------------#

config = Configuration() 

if __name__ == '__main__':

    model = TimmModel(config.model,
                          pretrained=True,
                          img_size=config.img_size)
                          
    data_config = model.get_config()
    
    mean = data_config["mean"]
    std = data_config["std"]
    img_size = (config.img_size, config.img_size)
    
    print("Start from:", config.checkpoint_start)
    model_state_dict = torch.load(config.checkpoint_start)  
    model.load_state_dict(model_state_dict, strict=False)     
        
    model = model.to(config.device)

    val_transforms, train_sat_transforms, train_drone_transforms = get_transforms(img_size, mean=mean, std=std)
    now_id = 0

    all_tours_dir = os.listdir(data_dir)
    for tour_id in tqdm(all_tours_dir):
        tour_path = data_dir + tour_id
        output_tour_path = output_dir + tour_id
        os.makedirs(output_tour_path, exist_ok=True)
        
        for image_id in os.listdir(tour_path):
            now_id += 1

            image_path = tour_path + '/' + image_id
            output_path = output_tour_path + '/' + image_id[:-5] + '.pkl'
            
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = val_transforms(image=img)['image'].to(config.device, non_blocking=True).float()            
            img.unsqueeze_(0)                                                                           

            emb = model(img).squeeze(0)

            with open(output_path, "wb") as f:
                pickle.dump(emb.detach().cpu().numpy(), f, protocol=pickle.HIGHEST_PROTOCOL)
 
