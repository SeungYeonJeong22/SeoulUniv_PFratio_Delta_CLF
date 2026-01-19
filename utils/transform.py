import torchvision.transforms as T

def get_transform(type='basic'):
    # Define a series of transformations to be applied to the images
    
    # resize, to tensor, normalize
    if type=='basic':
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        
    return transform