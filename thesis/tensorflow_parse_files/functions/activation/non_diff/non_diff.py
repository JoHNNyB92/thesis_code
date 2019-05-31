from functions.activation.activation import activation

class non_diff(activation):
    def insert_in_annetto(self):
        super(non_diff, self).insert_in_annetto()

    def __init__(self,node):
        super(activation, self).__init__(node.get_name(),node)