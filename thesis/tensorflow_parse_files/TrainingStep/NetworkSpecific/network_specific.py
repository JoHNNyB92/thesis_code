from TrainingStep.training_step import training_step

class network_specific(training_step):

    def insert_in_annetto(self):
        print("EIMAI NETWORK SPECIFIC")
        super(network_specific, self).insert_in_annetto()

    def __init__(self,node):
        super(network_specific, self).__init__(node)