class file_training_session:
    def __init__(self,name,epoch,steps):
        self.name=name
        self.session_epoch=epoch
        self.steps=steps
        self.next_session=""

    def print(self):
        print("LOGGING:Session is ",self.name)
        print("LOGGING:With session epoch",self.session_epoch)
        print("LOGGING:With steps :")
        if self.next_session!="":
            print("LOGGING:Next session is ",self.next_session)
        for st in self.steps:
            print("LOGGING:Step:[",st.name,"]")
            st.print()

    def remove_unsupported_chars(self):
        self.name=self.name.replace("\\", "_").replace("/", "_").replace(" ","_")
        self.next_session=self.next_session.replace("\\", "_").replace("/", "_").replace(" ","_")
        for st in self.steps:
            st.remove_unsupported_chars()
