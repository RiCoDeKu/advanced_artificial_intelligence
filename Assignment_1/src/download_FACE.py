import os
import torch
import argparse
import subprocess
from torchvision import transforms
from utils.data_io import CSVBasedDataset


# List of attributes for CelebA dataset
CELEBA_ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
    'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry',
    'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 
    'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 
    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 
    'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 
    'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 
    'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]

# Function to convert CelebA dataset images and labels into tensors
def make_CelebA_tensors(input_dir="tinyCelebA", csv_file="image_list.csv"):
    dataset = CSVBasedDataset(
        dirname=input_dir,
        filename=os.path.join(input_dir, csv_file),
        items=['File Path', CELEBA_ATTRIBUTES],
        dtypes=['image', 'float'],
        img_transform=transforms.CenterCrop((128, 128))
    )
    data = [dataset[i] for i in range(len(dataset))]
    x = torch.cat([torch.unsqueeze(u, dim=0) for u, v in data], dim=0)
    y = torch.cat([torch.unsqueeze(v, dim=0) for u, v in data], dim=0)
    
    del dataset, data
    return x, y

# CelebA dataset download and processing function
def download_and_process_celeba(output_dir, celeba_url):

    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["wget", celeba_url, "-O", "tinyCelebA.tar.gz"], check=True)
    subprocess.run(["tar", "-zxf", "tinyCelebA.tar.gz"], check=True)
    subprocess.run(["rm", "-f", "tinyCelebA.tar.gz"], check=True)
    
    # Process training data
    image_tensor, label_tensor = make_CelebA_tensors(input_dir="tinyCelebA", csv_file="image_list.csv")
    torch.save(image_tensor, os.path.join(output_dir, 'tinyCelebA_train_images.pt'))
    torch.save(label_tensor, os.path.join(output_dir, 'tinyCelebA_train_labels.pt'))
    del image_tensor, label_tensor
    
    subprocess.run(["rm", "-fr", "tinyCelebA"], check=True)

def main():
    """
    Main function to handle command-line arguments and execute processing
    """
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description='Download and process CelebA dataset')
    parser.add_argument('--data-dir', type=str, default="./data/CelebA")
    parser.add_argument('--celeba-url', type=str, default="https://tus.box.com/shared/static/z7a4pb9qtco6fwspige2tpt2ryhqv9l1.gz")
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    # Check if processed data already exists
    train_file = os.path.join(args.data_dir, 'tinyCelebA_train_images.pt')
    if not os.path.isfile(train_file) or args.force:
        download_and_process_celeba(args.data_dir, args.celeba_url)
    else:
        print(f"Processed data already exists in {args.data_dir}. Use --force to redownload.")

if __name__ == "__main__":
    main()