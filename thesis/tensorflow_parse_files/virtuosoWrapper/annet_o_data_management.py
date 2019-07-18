from virtuosoWrapper.crud import crud
from ANNConfiguration.ANNConfiguration import ANNConfiguration
from Network.network import network
from NetworkEvaluation.evaluation_result import evaluation_result


class annet_o_data_management:
    def __init__(self):
        self.prefix_syntax = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        self.prefix_owl="http://www.w3.org/2002/07/owl#"
        self.evaluationResult=""
        self.AnnConfig=""
        self.prefix_schema = "http://www.w3.org/2000/01/rdf-schema#"
        self.str_annet_0 = "annet_0"
        self.prefix_annet_o = "http://w3id.org/annett-o/"
        self.str_w3 = "w3"
        self.crud = crud()

    def init_data(self,name):
        self.AnnConfig=name
        self.annConfiguration=ANNConfiguration(name)

    def init_new_network(self,name):
        self.annConfiguration.networks[name]=network(name)

    def init_new_network_evaluation(self,name,network):
        self.evaluationResult = evaluation_result(name)
        self.evaluationResult.network=network

    def insert_annConfiguration(self,name):
        #Handle existing networks differently,depending on which network we have
        s=self.prefix_annet_o+name
        o=self.prefix_syntax+"type"
        p=self.prefix_annet_o+"ANNConfiguration"
        self.crud.insert(s,o,p)


    def insert_network(self,name):
        s = self.prefix_annet_o + self.annConfiguration.name
        o = self.prefix_annet_o + "hasNetwork"
        p = self.prefix_annet_o + name
        #TODO IF ANOTHER NETWORKS EXISTS>CHECK
        self.crud.insert(s, o, p)
        #return ann_config

    def insert_training_strategy(self,name):
        s = self.prefix_annet_o + self.annConfiguration.name
        o = self.prefix_annet_o + "hasTrainingStrategy"
        p = self.prefix_annet_o + name
        self.crud.insert(s, o, p)

    def insert_subclass(self,name,subclass):
        s=self.prefix_annet_o+name
        o = self.prefix_schema + "subClassOf"
        p=self.prefix_annet_o+subclass
        self.crud.insert(s, o, p)

    def insert_named_individual(self,name):
        s = self.prefix_annet_o + name
        o = self.prefix_owl + " NamedIndividual"
        p = self.prefix_annet_o + name
        self.crud.insert(s, o, p)

    def insert_type(self,name,type):
        s=self.prefix_annet_o+name
        o = self.prefix_syntax + "type"
        p=self.prefix_annet_o+type
        self.crud.insert(s, o, p)

    def insert_hasLayer(self,network,layer):
        s = self.prefix_annet_o + network
        o = self.prefix_annet_o + "hasLayer"
        p = self.prefix_annet_o + layer
        self.crud.insert(s, o, p)

    def insert_evaluatesAnnConfig(self,result):
        s = self.prefix_annet_o + result
        o = self.prefix_annet_o + "evaluatesANNConf"
        p = self.prefix_annet_o + self.AnnConfig
        self.crud.insert(s, o, p)

    def insert_evaluatesNetwork(self,result,network):
        s = self.prefix_annet_o + result
        o = self.prefix_annet_o + "evaluatesNetwork"
        p = self.prefix_annet_o + network
        self.crud.insert(s, o, p)

    def insert_EvaluatesUsingIOPipe(self, result, ioPipe):
        s = self.prefix_annet_o + result
        o = self.prefix_annet_o + "evaluatesUsingIOPipe"
        p = self.prefix_annet_o + ioPipe
        self.crud.insert(s, o, p)

    #def withTrainStrategy(self):

    def insert_joinsLayer(self,name,input):
        s = self.prefix_annet_o + name
        o = self.prefix_annet_o + "joinsLayer"
        p = self.prefix_annet_o + input
        self.crud.insert(s, o, p)

    def insert_hasMetric(self,result,metric):
        s = self.prefix_annet_o +result
        o = self.prefix_annet_o + "hasMetric"
        p = self.prefix_annet_o + metric
        self.crud.insert(s, o, p)

    def insert_hasActivation(self,name,activation):
        s = self.prefix_annet_o + name
        o = self.prefix_annet_o + "hasActivationFunction"
        p = self.prefix_annet_o + activation
        self.crud.insert(s, o, p)

    def insert_nextLayer(self,name,nextLayer):
        s = self.prefix_annet_o + name
        o = self.prefix_annet_o + "nextLayer"
        p = self.prefix_annet_o + nextLayer
        self.crud.insert(s, o, p)

    def insert_prevLayer(self,name,previousLayer):
        s = self.prefix_annet_o + name
        o = self.prefix_annet_o + "previousLayer"
        p = self.prefix_annet_o + previousLayer
        self.crud.insert(s, o, p)

    def insert_has_bias(self,layer,bool):
        value=str(bool)#+"^^<http://www.w3.org/2001/XMLSchema#boolean>"
        s = self.prefix_annet_o + layer
        o = self.prefix_annet_o + "has_bias"
        p = self.prefix_annet_o + value
        self.crud.insert(s, o, p)

    def insert_num_of_units(self,layer,num):
        s = self.prefix_annet_o + layer
        o = self.prefix_annet_o + "layer_num_units"
        p = self.prefix_annet_o + str(num)
        self.crud.insert(s, o, p)

    def insert_has_layer(self,network,layer):
        s = self.prefix_annet_o + network
        o = self.prefix_annet_o + "hasLayer"
        p = self.prefix_annet_o + layer
        self.crud.insert(s, o, p)

    def insert_has_loss(self,cost,loss):
        s = self.prefix_annet_o + cost
        o = self.prefix_annet_o + "hasLoss"
        p = self.prefix_annet_o + loss
        self.crud.insert(s, o, p)

    def insert_has_cost(self,objective,cost):
        s = self.prefix_annet_o + objective
        o = self.prefix_annet_o + "hasCost"
        p = self.prefix_annet_o + cost
        self.crud.insert(s, o, p)

    def insert_has_regularizer(self,cost,regularizer):
        s = self.prefix_annet_o + cost
        o = self.prefix_annet_o + "hasRegularizer"
        p = self.prefix_annet_o + regularizer
        self.crud.insert(s, o, p)

    def insert_has_objective(self,network,objective):
        s = self.prefix_annet_o + network
        o = self.prefix_annet_o + "hasObjective"
        p = self.prefix_annet_o + objective
        self.crud.insert(s, o, p)

    def insert_trains_network(self,training_, network):
        s = self.prefix_annet_o + training_
        o = self.prefix_annet_o + "trainsNetwork"
        p = self.prefix_annet_o + network
        self.crud.insert(s, o, p)

    def insert_io_pipe(self,training_,io_pipe):
        s = self.prefix_annet_o + training_
        o = self.prefix_annet_o + "trainingSingleHasIOPipe"
        p = self.prefix_annet_o + io_pipe
        self.crud.insert(s, o, p)

    def insert_joins_dataset(self,pipe,dataset):
        s = self.prefix_annet_o + pipe
        o = self.prefix_annet_o + "joinsDataset"
        p = self.prefix_annet_o + dataset
        self.crud.insert(s, o, p)

    def insert_joins_layer(self,pipe,layer):
        s = self.prefix_annet_o + pipe
        o = self.prefix_annet_o + "joinsLayer"
        p = self.prefix_annet_o + layer
        self.crud.insert(s, o, p)

    def insert_learning_rate(self,optimizer,learning_rate):
        s = self.prefix_annet_o + optimizer
        o = self.prefix_annet_o + "learning_rate"
        p = self.prefix_annet_o + str(learning_rate)
        self.crud.insert(s, o, p)

    def insert_optimizer(self,trStep,optimizer):
        s = self.prefix_annet_o + trStep
        o = self.prefix_annet_o + "hasTrainingOptimizer"
        p = self.prefix_annet_o + str(optimizer)
        self.crud.insert(s, o, p)

    def insert_trained_in(self,weight,trained_in):
        s = self.prefix_annet_o + weight
        o = self.prefix_annet_o + "trainedInLayer"
        p = self.prefix_annet_o + str(trained_in)
        self.crud.insert(s, o, p)

    def insert_trained_out(self,weight,trained_out):
        s = self.prefix_annet_o + weight
        o = self.prefix_annet_o + "trainedOutLayer"
        p = self.prefix_annet_o + str(trained_out)
        self.crud.insert(s, o, p)

    def insert_has_weight(self,model,weight):
        s = self.prefix_annet_o + model
        o = self.prefix_annet_o + "hasWeights"
        p = self.prefix_annet_o + str(weight)
        self.crud.insert(s, o, p)

    def insert_with_tr_str(self,evRes,tr):
        s = self.prefix_annet_o + evRes
        o = self.prefix_annet_o + "withTrainingStrategy"
        p = self.prefix_annet_o + str(tr)
        self.crud.insert(s, o, p)
    def insert_has_metric(self,evRes,metric):
        s = self.prefix_annet_o + evRes
        o = self.prefix_annet_o + "hasMetric"
        p = self.prefix_annet_o + str(metric)
        self.crud.insert(s, o, p)
    def insert_eval_using_io(self,evRes,pipe):
        s = self.prefix_annet_o + evRes
        o = self.prefix_annet_o + "evaluatesUsingIOPipe"
        p = self.prefix_annet_o + str(pipe)
        self.crud.insert(s, o, p)
    def insert_eval_network(self,evRes,network):
        s = self.prefix_annet_o + evRes
        o = self.prefix_annet_o + "evaluatesNetwork"
        p = self.prefix_annet_o + str(network)
        self.crud.insert(s, o, p)
    def insert_eval_ann_conf(self,evRes,ann_conf):
        s = self.prefix_annet_o + evRes
        o = self.prefix_annet_o + "hasPrimaryTrainingSession"
        p = self.prefix_annet_o + str(ann_conf)
        self.crud.insert(s, o, p)

    def insert_has_prim_tr_session(self,trStrat,sess):
        s = self.prefix_annet_o + trStrat
        o = self.prefix_annet_o + "hasPrimaryTrainingSession"
        p = self.prefix_annet_o + str(sess)
        self.crud.insert(s, o, p)
    def insert_has_trained_model(self,trStrat,model):
        s = self.prefix_annet_o + trStrat
        o = self.prefix_annet_o + "hasTrainedModel"
        p = self.prefix_annet_o + str(model)
        self.crud.insert(s, o, p)

    def insert_batch(self,annConf,batch):
        value = str(batch)# + "^^<http://www.w3.org/2001/XMLSchema#int>"
        s = self.prefix_annet_o + annConf
        o = self.prefix_annet_o + "batch_size"
        p = self.prefix_annet_o + str(value)
        self.crud.insert(s, o, p)

    def insert_epoch(self,annConf,epoch):
        value = str(epoch)# + "^^<http://www.w3.org/2001/XMLSchema#int>"
        s = self.prefix_annet_o + annConf
        o = self.prefix_annet_o + "number_of_epochs"
        p = self.prefix_annet_o + str(value)
        self.crud.insert(s, o, p)

    def insert_named_indiv(self,name):
        s = self.prefix_annet_o + name
        o = self.prefix_syntax + "type"
        p = self.prefix_owl + "NamedIndividual"
        self.crud.insert(s, o, p)

    def insert_has_network(self,network):
        s = self.prefix_annet_o + self.AnnConfig
        o = self.prefix_annet_o + "hasNetwork"
        p = self.prefix_owl + network
        self.crud.insert(s, o, p)

    def insert_has_training_strategy(self,trStrat):
        s = self.prefix_annet_o + self.AnnConfig
        o = self.prefix_annet_o + "hasTrainingStrategy"
        p = self.prefix_annet_o + trStrat
        self.crud.insert(s, o, p)

    def insert_input_layer(self,network,input):
        s = self.prefix_annet_o + network
        o = self.prefix_annet_o + "hasInputLayer"
        p = self.prefix_annet_o + input
        self.crud.insert(s, o, p)

    def insert_output_layer(self,network,output):
        s = self.prefix_annet_o + network
        o = self.prefix_annet_o + "hasOutputLayer"
        p = self.prefix_annet_o + output
        self.crud.insert(s, o, p)

    def insert_step(self,session,step):
        s = self.prefix_annet_o + session
        o = self.prefix_annet_o + "hasTrainingStep"
        p = self.prefix_annet_o + step
        self.crud.insert(s, o, p)

    def insert_primary_step(self,session,step):
        s = self.prefix_annet_o + session
        o = self.prefix_annet_o + "hasPrimaryTrainingStep"
        p = self.prefix_annet_o + step
        self.crud.insert(s, o, p)

    def insert_same_layer(self,layer,same_layer):
        s = self.prefix_annet_o + layer
        o = self.prefix_annet_o + "sameLayerAs"
        p = self.prefix_annet_o + same_layer
        self.crud.insert(s, o, p)

    def insert_has_stop_cond(self,loop,cond):
        s = self.prefix_annet_o + loop
        o = self.prefix_annet_o + "hasStopCondition"
        p = self.prefix_annet_o + cond
        self.crud.insert(s, o, p)

    def insert_stop_condition_number(self,cond,number):
        s = self.prefix_annet_o + cond
        o = self.prefix_annet_o + "iteratesFor"
        p = self.prefix_annet_o + str(number)
        self.crud.insert(s, o, p)

    def insert_primary_loop(self,loop,prLoopStep):
        s = self.prefix_annet_o + loop
        o = self.prefix_annet_o + "hasPrimaryLoopStep"
        p = self.prefix_annet_o + prLoopStep
        self.crud.insert(s, o, p)

    def insert_looping_step(self,loop,loopStep):
        s = self.prefix_annet_o + loop
        o = self.prefix_annet_o + "hasLoopTrainingStep"
        p = self.prefix_annet_o + loopStep
        self.crud.insert(s, o, p)