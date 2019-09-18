import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class iterates_for:

    def insert_in_annetto(self):
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        rdfWrapper.new_num_of_iterations(self.name, self.num_of_iterations)

    def __init__(self,name,value):
        self.type="IterateFor"
        self.name=name
        self.num_of_iterations=value