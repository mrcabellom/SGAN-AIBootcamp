
from model.srgan import SRGAN


if __name__ == '__main__':
    gan = SRGAN(dataset='images_test')
    gan.load_model_weights('saved_model/model.h5')
    gan.eval_images(1,batch_size=None)