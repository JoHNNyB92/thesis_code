class file_tr_step:
    def __init__(self):
        self.name=""
        self.loss=[]
        self.optimizer=[]
        self.epoch=-1
        self.inputs=[]
        self.batches=[]
        self.objective=""
        self.next=""
        self.network=""
        self.next_file=""


    def print(self):
        print("Name=",self.name)
        print("Inputs=", self.inputs)
        print("Batches=",self.batches)
        print("Loss=",self.loss)
        print("Optimizer=",self.optimizer)
        print("Epoch=",self.epoch)
        if self.objective!="":
            print("Objective=",self.objective.cost_function.loss)
        if self.next!="":
            print("Next Step is =",self.next)
        if self.next_file!="":
            print("Next File step is =",self.next_file)
        print("Network=",self.network)
