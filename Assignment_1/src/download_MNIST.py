import os
import torch
import argparse
import subprocess
from torchvision import transforms
from utils.data_io import CSVBasedDataset

# Function to convert MNIST dataset images and labels into tensors
def make_MNIST_tensors(csv_file, input_dir):
    transform = transforms.Grayscale()
    dataset = CSVBasedDataset(
        dirname=input_dir, 
        filename=os.path.join(input_dir, csv_file), 
        items=['File Path', 'Class Label'], 
        dtypes=['image', 'label']
    )
    data = [dataset[i] for i in range(len(dataset))]
    x = torch.cat([torch.unsqueeze(transform(u), dim=0) for u, v in data], dim=0)
    y = torch.cat([torch.unsqueeze(v, dim=0) for u, v in data], dim=0)
    del dataset, data
    return x, y

# MNIST dataset download and processing function
def download_and_process_mnist(output_dir, mnist_url):
    
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["wget", mnist_url, "-O", "MNIST.tar.gz"], check=True)
    subprocess.run(["tar", "-zxf", "MNIST.tar.gz"], check=True)
    subprocess.run(["rm", "-f", "MNIST.tar.gz"], check=True)
    
    # Process training data
    train_images, train_labels = make_MNIST_tensors(csv_file="train_list.csv", input_dir= "MNIST" )
    torch.save(train_images, os.path.join(output_dir, 'MNIST_train_images.pt'))
    torch.save(train_labels, os.path.join(output_dir, 'MNIST_train_labels.pt'))
    del train_images, train_labels
    
    # Process test data
    test_images, test_labels = make_MNIST_tensors(csv_file="test_list.csv", input_dir= "MNIST" )
    torch.save(test_images, os.path.join(output_dir, 'MNIST_test_images.pt'))
    torch.save(test_labels, os.path.join(output_dir, 'MNIST_test_labels.pt'))
    del test_images, test_labels
    
    # Clean up extracted directory
    subprocess.run(["rm", "-fr",  "MNIST" ], check=True)

def main():
    parser = argparse.ArgumentParser(description='Download and process MNIST dataset')
    parser.add_argument('--data-dir', type=str, default="./data/MNIST")
    parser.add_argument('--mnist-url', type=str, default="https://tus.box.com/shared/static/98etdh5hh5bourjs6izjkkxvjw5tjf1w.gz")
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    # Check if processed data already exists
    train_file = os.path.join(args.data_dir, 'MNIST_train_images.pt')
    if not os.path.isfile(train_file) or args.force:
        download_and_process_mnist(args.data_dir, args.mnist_url)
    else:
        print(f"Processed data already exists in {args.data_dir}. Use --force to redownload.")

if __name__ == "__main__":
    main()