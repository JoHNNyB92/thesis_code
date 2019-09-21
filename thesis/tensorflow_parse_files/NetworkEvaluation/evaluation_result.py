from NetworkEvaluation.network_evaluation import network_evaluation
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
import nodes.handler
class evaluation_result(network_evaluation):

    def insert_in_annetto(self):
        if self.metric!="":
            rdfWrapper.new_named_individual(self.name)
            rdfWrapper.new_type(self.name,self.type)
            rdfWrapper.new_evaluates_ann_conf(self.name, self.ann_conf.name)
            for i,_ in enumerate(self.IOPipe):
                self.IOPipe[i].insert_in_annetto()
                rdfWrapper.new_evaluates_using_io(self.name,self.IOPipe[i].name)
            self.metric.insert_in_annetto()
            rdfWrapper.new_has_metric(self.name, self.metric.name)
            rdfWrapper.new_evaluates_network(self.name, self.network)
            rdfWrapper.new_with_tr_strategy(self.name, self.train_strategy.name)
        else:
            print("ERROR:Neural network w/o evaluation.")

    def __init__(self,name):
        self.name=name
        self.type="EvaluationResult"
        super(evaluation_result, self).__init__()
        self.eval_score=0
        self.ann_conf=""
        self.network=""
        self.IOPipe=""
        self.train_strategy=""
        self.metric=""

        eval_score=0





