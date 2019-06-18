from functions.activation.non_diff.non_diff import non_diff

class elu(non_diff):

    def insert_in_annetto(self):
        super(elu, self).insert_in_annetto()

    def __init__(self,node):
        super(elu, self).__init__(node)
        self.type="Elu"