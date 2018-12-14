
from model.srgan import SRGAN

if __name__ == '__main__':
    gan = SRGAN(dataset='img_align_celeba')
    gan.train(epochs=30000, batch_size=1, sample_interval=50)
