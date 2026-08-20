import matplotlib.pyplot as plt

class Circle(object):
    def __init__(self, r= 3, c= 'blue'):
        self.r= r
        self.c= c
    def add_r (self, r):
        self.r = self.r + r
        return(self.r)
    def draw(self):
        plt.gca().add_patch(plt.Circle((0,0), radius=self.r, fc=self.c))
        plt.axis('scaled')
        plt.shpw

redcircle= Circle(10, 'red')
print(dir(redcircle),
redcircle.r, redcircle.c)
redcircle.draw
