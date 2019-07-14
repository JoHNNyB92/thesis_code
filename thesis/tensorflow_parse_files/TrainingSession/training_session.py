import virtuosoWrapper.virtuosoWrapper as rdfWrapper
from  TrainingStep.TrainingLoop.training_loop import training_loop
class training_session:

    def insert_in_annetto(self):
        #print("Annetto::training_session-", self.name)
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)

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


    def __init__(self,name,trStep):
        self.type="TrainingSession"
        self.name=name
        primaryInLoop=""
        loopSteps=[]
        print("ULTIMATETEST:Training Session Info:",self.name)
        for trainingStep in trStep:
            if trainingStep.isPrimaryInLoop==True:
                print("ULTIMATETEST:Primary In Loop:",trainingStep.name)
                print("ULTIMATETEST:Epochs:",trainingStep.epochs)
                primaryInLoop=trainingStep
            elif trainingStep.isLoopingStep==True:
                print("ULTIMATETEST:Loop Step:", trainingStep.name)
                print("ULTIMATETEST:Epochs:", trainingStep.epochs)
                loopSteps.append(trainingStep)
            else:
                print("ULTIMATETEST:Simple training step:", trainingStep.name)
                print("ULTIMATETEST:Epochs:", trainingStep.epochs)
                self.hasTrainingStep=trStep
        if len(trStep)==1:
            self.hasPrimaryTrainingStep=trStep[0]
        if primaryInLoop!="":
            tr_loop=training_loop(primaryInLoop.epochs,primaryInLoop,loopSteps)
            self.hasPrimaryTrainingStep=tr_loop
        print("ULTIMATETEST:Primary Step is:", trainingStep.name)
        print("ULTIMATETEST:Epochs:", trainingStep.epochs)
