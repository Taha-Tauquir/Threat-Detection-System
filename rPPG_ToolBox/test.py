import numpy as np

def x():
    input_path = "C:/Users/Hp/Downloads/MAHNOB-SAMPLE_DATASET/tst/552/input/Input_0001.npy"
    data = np.load(input_path)

    print("Shape:", data.shape)
    print("Type:", data.dtype)
    print("Min:", data.min(), "Max:", data.max())

if __name__ == '__main__':
    x()







