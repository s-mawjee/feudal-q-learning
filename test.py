import numpy as np
states = []
states.append((4, 4))
for i in range(3-2, -1, -1):
    states.append((int(states[-1][0]/2), int(states[-1][1]/2)))
states.reverse()
print(states)

actions = np.ones(3) * -1
print(actions)
