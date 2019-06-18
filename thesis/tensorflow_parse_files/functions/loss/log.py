from functions.loss.loss import loss

class log(loss):

    def insert_in_annetto(self):
        super(log, self).insert_in_annetto()

    def __init__(self,node):
        super(loss, self).__init__(node.get_name(), node)
        self.type="Log"