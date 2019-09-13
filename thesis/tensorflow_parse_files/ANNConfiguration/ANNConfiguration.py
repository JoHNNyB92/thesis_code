from Network.network import network
import nodes.handler
import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class ANNConfiguration:

    def insert_in_annetto(self):
        rdfWrapper.new_named_individual(self.name)
        for network in nodes.handler.entitiesHandler.data.annConfiguration.networks.keys():
            network_node = nodes.handler.entitiesHandler.data.annConfiguration.networks[network]
            network_node.insert_in_annetto_netwr()
        for trStrategy in nodes.handler.entitiesHandler.data.annConfiguration.training_strategy.keys():
            trStrategy_node = nodes.handler.entitiesHandler.data.annConfiguration.training_strategy[trStrategy]
            trStrategy_node.insert_in_annetto()
        rdfWrapper.new_type(self.name, self.type)
        rdfWrapper.new_ann_configuration(self.name)

    def __init__(self,name):
        self.name=name
        self.type="ANNConfiguration"
        self.networks = {}
        self.training_strategy = {}