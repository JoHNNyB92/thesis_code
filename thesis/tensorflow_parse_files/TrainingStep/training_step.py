import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class training_step:

    def insert_in_annetto(self):
        #print("Annetto::training_step-", self.name)
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        if self.nextTrStep!="":
            rdfWrapper.new_next_step(self.name,self.nextTrStep)
        for network in self.networks:
            rdfWrapper.new_trains_network(self.name,network)
        inserted=[]
        for i,_ in enumerate(self.IOPipe):
            if self.IOPipe[i].name not in inserted:
                inserted.append(self.IOPipe[i].name)
                self.IOPipe[i].insert_in_annetto()
                rdfWrapper.new_io_pipe(self.name,self.IOPipe[i].name)
        rdfWrapper.new_has_optimizer(self.name,self.trainingOptimizer)

    def __init__(self,name):
        self.name=name
        self.type="TraininStep"