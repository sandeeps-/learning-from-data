import numpy as np;

def coinFlip(nCoins, trials):
    #Create  nCoins each flipped trial times
    coinSamples = np.round(np.matrix(np.random.uniform(0, 1, [trials, nCoins])))
    v1 = np.sum(coinSamples[:, 0], axis=0)[0, 0] / trials
    vrand = np.sum(coinSamples[:, np.random.randint(0, nCoins)], axis=0)[0, 0] / trials
    vmin = np.min(np.sum(coinSamples, axis=0)) / trials
    return np.matrix([v1, vrand, vmin])

if __name__ == '__main__':
    count = np.matrix([0, 0 , 0])
    for i in range(0, 100000):
        count = count + coinFlip(1000, 10)
    
    v = count / 100000.0
    print("[v1, vrand, vmin] = %s" %v)    
