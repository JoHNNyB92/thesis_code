import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class training_step:

    def insert_in_annetto(self):
        print("Annetto::training_step-", self.name)
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        rdfWrapper.new_trains_network(self.name,self.network)
        for i,_ in enumerate(self.IOPipe):
            self.IOPipe[i].insert_in_annetto()
            rdfWrapper.new_io_pipe(self.name,self.IOPipe[i].name)
        for optimizer in self.trainingOptimizer:
            print("OPTIMIZER= ",optimizer)
            rdfWrapper.new_has_optimizer(self.name,optimizer)
        print("LEAVING=",self.name)

    def __init__(self,name):
        self.name=name
        self.type="TraininStep"