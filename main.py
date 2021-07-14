import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    l = torch.randint(10, (5000,))
    s = torch.rand((5000, 10))
    s = torch.softmax(s, 1)

    # df = pd.DataFrame()
    # df['labels'] = l.numpy()
    # df['beliefs'] = s.numpy()
    #
    sort_s = torch.sort(s, 1)

    mean = torch.mean(sort_s.values, 0)
    median = torch.median(sort_s.values, 0).values
    std = torch.std(sort_s.values, 0)
    l_max, h_min = sort_s.values[:, 0].max(), sort_s.values[:, 9].min()

    # plt.bar(range(len(mean)), mean)
    plt.boxplot(torch.transpose(sort_s.values, 0, 1), showfliers=False, whis=(0, 100))
    plt.axhline(y=l_max, color="red", linestyle=(0, (5, 5)))
    plt.axhline(y=h_min, color="blue", linestyle=(0, (5, 5)))

    plt.show()