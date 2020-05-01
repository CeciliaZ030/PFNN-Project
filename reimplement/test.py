import matplotlib.pyplot as plt
import re

with open('h.txt', 'r') as f:
    content = f.read().splitlines()

    index = []
    loss = []
    for (i, line) in enumerate(content):

        temp = re.findall('loss: (.+)$', line)

        if float(temp[0]) > 1.8: continue
        loss.append(float(temp[0]))

        temp = re.findall('^[0-9]+', line)
        index.append(int(temp[0]))



    # print(index)



    plt.plot(index, loss)
    plt.title('loss vs. iterations')
    plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.show()
