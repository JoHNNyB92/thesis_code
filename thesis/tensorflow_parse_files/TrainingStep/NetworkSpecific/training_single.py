from TrainingStep.NetworkSpecific.network_specific import network_specific
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
import nodes.handler

class training_single(network_specific):

    def insert_in_annetto(self):
        #rdfWrapper.new_named_individual(self.name)
        super(training_single, self).insert_in_annetto()
        #rdfWrapper.new_type(self.name, self.type)
        rdfWrapper.new_batch_size(self.name, self.batch)
        rdfWrapper.new_epoch_num(self.name, self.epochs)

    def find_learning_rate_decay(self):
        nm=nodes.handler.entitiesHandler.node_map
        for key in nm.keys():
            if "decay/cast_1" in nm[key].get_name().lower():
                self.learning_rate_decay=nm[key].node_obj.attr["value"].tensor.float_val
                break


    def __init__(self,name,network,IOPipe,trainingOptimizer,epochs,batch,nextTrStep):
        super(training_single, self).__init__(name)
        self.type="TrainingSingle"
        self.networks=network
        self.IOPipe=IOPipe
        self.nextTrStep=nextTrStep
        self.trainingOptimizer=trainingOptimizer
        self.epochs=epochs
        self.batch=batch
        self.find_learning_rate_decay()
        self.learning_rate_decay=""
        self.epoch_decay=""
        self.isInLoop=False
        self.isPrimaryInLoop=False
        self.isLoopingStep=False