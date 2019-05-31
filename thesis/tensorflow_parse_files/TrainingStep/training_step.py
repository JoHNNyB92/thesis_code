import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class training_step:

    def insert_in_annetto(self):
        print("Annetto::training_step-", self.name)
        rdfWrapper.new_trains_network(self.name,self.network)
        for i,_ in enumerate(self.IOPipe):
            self.IOPipe[i].insert_in_annetto()
            rdfWrapper.new_io_pipe(self.name,self.IOPipe[i].name)
        self.trainingOptimizer.insert_in_annetto()
        rdfWrapper.new_has_optimizer(self.name,self.trainingOptimizer.name)


    def __init__(self,name):
        self.name=name