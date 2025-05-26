import os
import shutil
import train_AE_MNIST as train_ae_mnist
import train_AE_CelebA as train_ae_celeb
import train_VAE_MNIST as train_vae_mnist
import train_VAE_CelebA as train_vae_celeb

def main():
	dim = [32,64,128,256]
	# MNISTのAEを学習
	for n in dim:
		train_ae_mnist.train(n)
	# CelebAのAEを学習
	for n in dim:
		train_ae_celeb.train(n)

	# MNISTのVAEを学習
	for n in dim:
		train_vae_mnist.train(n)

	# CelebAのVAEを学習
	for n in dim:
		train_vae_celeb.train(n)

if __name__ == '__main__':
	main()