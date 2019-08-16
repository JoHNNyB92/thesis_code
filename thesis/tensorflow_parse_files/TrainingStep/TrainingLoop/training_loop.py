import virtuosoWrapper.virtuosoWrapper as rdfWrapper
from StopCondition.iterates_for import iterates_for

class training_loop:

    def insert_in_annetto(self):
        rdfWrapper.new_named_individual(self.name)
        rdfWrapper.new_type(self.name, self.type)

        self.hasStopCondition.insert_in_annetto()
        rdfWrapper.new_has_stop_cond(self.name,self.hasStopCondition.name)
        self.primaryLoop.insert_in_annetto()
        print("done=",self.primaryLoop.name)
        if self.primaryLoop!="":
            rdfWrapper.new_has_primary_loop(self.name,self.primaryLoop.name)
        else:
            print("LOGGING:There is no primary loop.")
        for lstep in self.loopSteps:
            print("volta=",lstep)
            lstep.insert_in_annetto()
            rdfWrapper.new_has_looping_step(self.name, lstep.name)

    def __init__(self,name,primaryLoop,loopSteps,stopCond):
        self.type="TrainingLoop"
        self.name=name
        self.hasStopCondition=iterates_for(self.name+"_iterates_for",stopCond)
        self.primaryLoop=primaryLoop
        self.loopSteps=loopSteps