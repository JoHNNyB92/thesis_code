from functions.loss.loss import loss

class mse(loss):

    def insert_in_annetto(self):
        super(mse, self).insert_in_annetto()

    def __init__(self,node,name):
        super(mse, self).__init__(name, node)
        self.type="MSE"