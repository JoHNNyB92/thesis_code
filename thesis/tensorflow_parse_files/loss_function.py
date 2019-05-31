class loss_function:
    def __init__(self,Y,Prod_Y):
        self.Y=Y
        self.P_Y=Prod_Y
        print("Real Y is ",self.Y.name," and prod ",self.P_Y.name)

