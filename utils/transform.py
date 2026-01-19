import torchvision.transforms as T

def get_transform(cfg=None):
    # Define a series of transformations to be applied to the images
    
    if cfg is None:
        cfg = {
            "transform":{
                "image_size": 224   
            }
        }
    
    # resize, to tensor, normalize
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((cfg.transform["image_size"], cfg.transform["image_size"])),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
        
    return transform