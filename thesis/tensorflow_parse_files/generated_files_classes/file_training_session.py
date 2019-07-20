class file_training_session:
    def __init__(self,name,epoch,steps):
        self.name=name
        self.session_epoch=epoch
        self.steps=steps

    def print(self):
        print("Session is ",self.name)
        print("With session epoch",self.session_epoch)
        print("With steps :")
        for st in self.steps:
            print("Step:[",st.name,"]")
            st.print()