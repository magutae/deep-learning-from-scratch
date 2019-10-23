class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized")
    
    def hello(self):
        print("Hello " + self.name)

    def goodboy(self):
        print("Good-bye " + self.name)


m = Man("익태")
m.hello()
m.goodboy()
