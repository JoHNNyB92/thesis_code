from functions.RegularizerFunction.regularizer_function import regularizer_function

class l2regularization(regularizer_function):

    def insert_in_annetto(self):
        super(l2regularization, self).insert_in_annetto()

    def __init__(self,node):
        self.type="L2Regularization"
        super(l2regularization, self).__init__(node.get_name(),node)