from functions.metric.metric import metric

class accuracy(metric):

    def insert_in_annetto(self):
        super(accuracy, self).insert_in_annetto()

    def __init__(self,node):
        self.type="Accuracy"
        self.metric=node.get_name()
        super(accuracy, self).__init__(node.get_name(),node)