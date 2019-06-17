from functions.activation.smooth.smooth import smooth

class sigmoid(smooth):

    def insert_in_annetto(self):
        super(sigmoid, self).insert_in_annetto()

    def __init__(self,node):
        super(sigmoid, self).__init__(node)
        self.type="Sigmoid"