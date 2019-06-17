from functions.activation.smooth.smooth import smooth

class softplus(smooth):

    def insert_in_annetto(self):
        super(softplus, self).insert_in_annetto()

    def __init__(self,node):
        super(softplus, self).__init__(node)
        self.type="Softplus"