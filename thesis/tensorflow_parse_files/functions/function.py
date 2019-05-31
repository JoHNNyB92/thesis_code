import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class function:

    def insert_in_annetto(self):
        print("Annetto::function-", self.name)
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)

    def __init__(self,name,node):
        self.name=name
        self.node=node