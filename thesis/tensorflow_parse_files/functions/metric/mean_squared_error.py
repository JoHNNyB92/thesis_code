from functions.metric.metric import metric

class mean_squared_error(metric):

    def insert_in_annetto(self):
        super(mean_squared_error, self).insert_in_annetto()

    def __init__(self,node):
        self.type="MeanSquaredError"
        self.metric=node.get_name()
        super(mean_squared_error, self).__init__(node.get_name(),node)