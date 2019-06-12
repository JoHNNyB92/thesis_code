import virtuosoWrapper.virtuosoWrapper as rdfWrapper

class training_session:

    def insert_in_annetto(self):
        #print("Annetto::training_session-", self.name)
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        if self.hasTrainingStep!="":
            for tr_step in self.hasTrainingStep:
                tr_step.insert_in_annetto()
                rdfWrapper.new_has_training_step(self.name,tr_step.name)
        else:
            print("ERROR:TrainingStep is empty")

        if self.hasPrimaryTrainingStep!="":
            self.hasPrimaryTrainingStep.insert_in_annetto()
            rdfWrapper.new_has_primary_training_step(self.name,self.hasPrimaryTrainingStep.name)
        else:
            print("ERROR:PrimaryTrainingStep is empty")


    def __init__(self,name,trStep,primaryTrStep):
        self.type="TrainingSession"
        self.name=name
        self.hasTrainingStep=trStep
        self.hasPrimaryTrainingStep=primaryTrStep


