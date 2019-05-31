import virtuosoWrapper.virtuosoWrapper as rdfWrapper
class training_strategy:

    def insert_in_annetto(self):
        print("Annetto::TrainingStrategy-",self.name)
        self.primary_training_session.insert_in_annetto()
        self.training_model.insert_in_annetto()
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        rdfWrapper.new_training_strategy(self.name)
        rdfWrapper.new_has_prim_tr_session(self.name,self.primary_training_session.name)
        rdfWrapper.new_trained_model(self.name,self.training_model.name)


    def __init__(self,name,session,trModel):
        self.name=name
        self.type="TrainingStrategy"
        self.primary_training_session=session
        self.training_model=trModel