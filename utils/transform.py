import torchvision.transforms as T
import monai.transforms as MT

def get_transform(cfg=None):
    # Define a series of transformations to be applied to the images
    
    if cfg is None:
        cfg = {
            "transform":{
                "image_size": 224   
            }
        }
    
    # # resize, to tensor, normalize
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((cfg.transform.image_size, cfg.transform.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # # monai transform example
    # transform = MT.Compose([
    #     MT.EnsureChannelFirst(channel_dim="no_channel"), # 기본적으로 monai에서는 뒤의 2차원을 바꾸기 때문에 채널이 고정돼있다느 사실을 알려야 함
    #     MT.Resize((cfg.transform.image_size, cfg.transform.image_size)),
    #     MT.ScaleIntensity(),
    #     MT.ToTensor()
    # ])
        
    return transform