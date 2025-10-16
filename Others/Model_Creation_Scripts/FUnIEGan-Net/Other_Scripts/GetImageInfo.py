import sys
import cv2

if __name__ == "__main__":
    imageFile = "/home/ashwin/Project/Original/FUnIE-GAN/TF-Keras/checkpoints/UGAN/wgan_pix2pix_underwater_imagenet/samples/old2/1000_gen.png"
    inputImage = None

    try:
        inputImage = cv2.imread(imageFile, -1)
        print(inputImage)
        if inputImage is None:
            print("Could not load image.")
            exit()
    except Exception as e:
        print("Exception - ", str(e))
        exit()
    imgHeight, imgWidth, numChannels = inputImage.shape
    print(f"{imageFile}: height={imgHeight}, width={imgWidth}, channels={numChannels}")
exit()
