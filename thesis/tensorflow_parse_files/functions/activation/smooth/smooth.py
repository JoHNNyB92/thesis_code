from functions.activation.activation import activation

class smooth(activation):

    def insert_in_annetto(self):
        super(smooth, self).insert_in_annetto()

    def __init__(self, node):
        super(smooth, self).__init__(node.get_name(), node)