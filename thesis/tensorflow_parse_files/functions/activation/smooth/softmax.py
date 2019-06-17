from functions.activation.smooth.smooth import smooth

class softmax(smooth):

    def insert_in_annetto(self):
        super(softmax, self).insert_in_annetto()

    def __init__(self,node):
        super(softmax, self).__init__(node)
        self.type="Softmax"