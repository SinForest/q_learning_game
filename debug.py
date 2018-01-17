import numpy as np

class Debugger:

    def __init__(self, fp, agent):
        self.screens = np.load(fp)
        self.agent   = agent
        for i, sc1 in enumerate(self.screens):
            for sc2 in self.screens[:i]:
                if (sc1 == sc2).all():
                    print("There are identical screens in the debug data.")
    
    def eval_screens(self):
        X = self.agent.to_var(self.screens)
        pred = self.agent.model(X, verbose=True)
        print("debug screens:", pred)
        X = self.agent.to_var(np.random.randint(0,256, self.screens.shape))
        pred = self.agent.model(X, verbose=True)
        print("sanity:", pred)