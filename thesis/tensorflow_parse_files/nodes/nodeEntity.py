class nodeEntity:
    def __init__(self,obj,_inputs,name=""):
        self.node_obj=obj
        self.inputs=_inputs
        if name!="":
            self.node_obj.name=name

    def get_attr(self):
        return self.node_obj.attr

    def get_output(self):
        return self.node_obj.attr["_output_shapes"].list.shape

    def get_op(self):
        return self.node_obj.op

    def get_name(self):
        return self.node_obj.name

    def get_inputs(self):
        return self.inputs

    def search_inputs(self,name):
        for elem in self.inputs:
            if elem.get_name()==name:
                return True

    def search_name_part_in_inputs(self, name):
        for elem in self.inputs:
            if name in elem.get_name():
                return True

    def get_specific_type(self,type):
        for elem in self.inputs:
            if elem.get_op()==type:
                return elem.get_name()


