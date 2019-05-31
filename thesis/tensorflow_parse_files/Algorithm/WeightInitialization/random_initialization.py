from Algorithm.WeightInitialization.weight_initialization import weight_initialization

class random_initialization(weight_initialization):

    def __init__(self,node):
        super(random_initialization, self).__init__(node, node.get_name())
        self.type="NormalInitialization"
