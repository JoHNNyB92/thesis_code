import virtuosoWrapper.virtuosoWrapper as rdfWrapper
from  TrainingStep.TrainingLoop.training_loop import training_loop

class training_session:
    def insert_in_annetto(self):
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)
        print("ALOKOTA PRAMATA")
        if self.hasTrainingStep!="":
            for ind,tr_step in enumerate(self.hasTrainingStep):
                tr_step.insert_in_annetto()
                rdfWrapper.new_has_training_step(self.name,tr_step.name)
        else:
            print("ERROR:TrainingStep is empty")

        if self.hasPrimaryTrainingStep!="":
            self.hasPrimaryTrainingStep.insert_in_annetto()
            rdfWrapper.new_has_primary_training_step(self.name,self.hasPrimaryTrainingStep.name)
        else:
            print("ERROR:PrimaryTrainingStep is empty")


    def __init__(self,name,trSteps,primaryTrainingStep):
        self.type="TrainingSession"
        self.name=name
        self.hasTrainingStep=[]
        self.hasPrimaryTrainingStep=primaryTrainingStep

        if "training_single" in str(self.hasPrimaryTrainingStep.__class__):
            print("LOGGING:Training:Primary training step:", self.hasPrimaryTrainingStep.name)
            print("LOGGING:Training:Primary training step epochs:", self.hasPrimaryTrainingStep.epochs)
            print("LOGGING:Primary training step batch:", self.hasPrimaryTrainingStep.batch)
        elif self.hasPrimaryTrainingStep!="":
            print("LOGGING:Training:TrainingLoop", self.hasPrimaryTrainingStep.name)
            print("LOGGING:Primary TrainingLoop epochs:", self.hasPrimaryTrainingStep.primaryLoop.epochs)
            print("LOGGING:Primary TrainingLoop batch:", self.hasPrimaryTrainingStep.primaryLoop.batch)
            for trainingStep in self.hasPrimaryTrainingStep.loopSteps:
                print("LOGGING:Training:Looping training step:", trainingStep.name)
                print("LOGGING:Training:Looping Step epochs:", trainingStep.epochs)
                print("LOGGING:Training:Looping Step batch:", trainingStep.batch)

        for trainingStep in trSteps:
            self.hasTrainingStep.append(trainingStep)
            print("LOGGING:Training:Simple training step:", trainingStep.name)
            print("LOGGING:Training:Step epochs:", trainingStep.epochs)
            print("LOGGING:Training:Step batch:", trainingStep.batch)
