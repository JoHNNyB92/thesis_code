from functions.activation.regularization.regularization import regularization

class dropout(regularization):

    def insert_in_annetto(self):
        super(dropout, self).insert_in_annetto()

    def non_diff(self,node):
        super(dropout, self).__init__(node.get_name(),node)
        self.type="Dropout"