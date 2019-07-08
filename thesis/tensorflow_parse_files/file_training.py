class file_training:
    def __init__(self):
        self.name=""
        self.loss=[]
        self.optimizer=[]
        self.epoch=-1
        self.inputs=[]

    def print(self):
        print("Name=",self.name)
        print("Inputs=", self.inputs)
        print("Loss=",self.loss)
        print("Optimizer=",self.optimizer)
        print("Epoch=",self.epoch)