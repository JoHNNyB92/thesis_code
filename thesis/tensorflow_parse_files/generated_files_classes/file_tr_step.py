
'''
Class that contains information retrieved from executing the neural network main python file for each training step.
File contains extra commands to print those information into a different file.
'''

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

    #Change names that contain the folder separator(\\) with _.
    def remove_unsupported_chars(self):
        self.name=self.name.replace("\\","_").replace("/","_").replace(" ","_")
        self.next=self.next.replace("\\","_").replace("/","_").replace(" ","_")

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