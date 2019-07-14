class training_loop:



    def __init__(self,stopCond,primaryLoop,loopSteps):
        self.hasStopCondition=stopCond
        self.primaryLoop=primaryLoop
        self.loopSteps=loopSteps