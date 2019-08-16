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
        print("--------------------------------------------------")
        print("LOGGING:Name=",self.name)
        print("LOGGING:Inputs=", self.inputs)
        print("LOGGING:Batches=",self.batches)
        print("LOGGING:Loss=",self.loss)
        print("LOGGING:Optimizer=",self.optimizer)
        print("LOGGING:Epoch=",self.epoch)
        if self.objective!="":
            print("LOGGING:Objective=",self.objective.cost_function.loss)
        if self.next!="":
            print("LOGGING:Next Step is =",self.next)
        if self.next_file!="":
            print("LOGGING:Next File step is =",self.next_file)
        print("LOGGING:Network=",self.network)
        print("--------------------------------------------------")