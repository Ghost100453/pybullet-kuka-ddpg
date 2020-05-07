from tqdm import tqdm
import math

def main():
    pi = 0
    for i in tqdm(range(int(1e9))):
        r = math.pow(-1, i)*1./(2*i+1)
        pi += r
    return pi*4

if __name__ == '__main__':
    pi = main()
    print('pi:', pi)

