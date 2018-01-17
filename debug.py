import numpy as np

class Debugger:

    def __init__(self, fp, agent):
        self.screens = np.load(fp)
        self.agent   = agent
    
    def eval_screens(self):
        X = self.agent.to_var(self.screens)
        pred = self.agent.model(X)
        print(pred)
        """
        for i in range(len(pred)):
            print(pred.data[i])
        """