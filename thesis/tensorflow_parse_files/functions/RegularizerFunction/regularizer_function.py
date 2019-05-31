from functions.function import function

class regularizer_function(function):

    def insert_in_annetto(self):
        super(regularizer_function, self).insert_in_annetto()

    def __init__(self,name,node):
        super(regularizer_function, self).__init__(name,node)
