from functions.activation.non_diff.non_diff import non_diff

class relu(non_diff):

    def insert_in_annetto(self):
        super(relu, self).insert_in_annetto()

    def __init__(self,node):
        super(relu, self).__init__(node)
        self.type="Relu"