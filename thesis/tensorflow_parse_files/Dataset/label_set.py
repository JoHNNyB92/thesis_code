import nodes.handler
import virtuosoWrapper.virtuosoWrapper as rdfWrapper
from Dataset.dataset import dataset

class label_set(dataset):

    def insert_in_annetto(self):
        res=rdfWrapper.new_named_individual(self.name)
        if res==0:
            rdfWrapper.new_type(self.name, self.type)

    def __init__(self,node):
        super(label_set, self).__init__(node)
        self.count=0
        self.name=node.get_name()+"_label_set"
        self.type="Labelset"
        for elem in self.node.get_output():
            for num in elem.dim:
                print(
                '''
                We want to take the number of labels.With that,we want only the last dimension size.
                Thus,we iterate and we assign the last one.
                '''
                )
                if int(num.size) > 0:
                    self.count=int(num.size)
        print("LOGGIND:Found size", self.count)