from functions.activation.smooth.smooth import smooth

class tanh(smooth):
    def insert_in_annetto(self):
        super(tanh, self).insert_in_annetto()

    def __init__(self,node):
        super(tanh, self).__init__(node)
        self.type="Tanh"