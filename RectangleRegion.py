
class RectangleRegion:

    def __init__(self,y,x,width,height):
        self.x = x 
        self.y = y 
        self.height = height 
        self.width = width 

    def compute_feature(self,ii):
        BR = ii[self.y-1+self.height-1,self.x+self.width-1]
        TL = ii[self.y-1,self.x-1] if self.y>0 and self.x>0 else 0 
        TR = ii[self.y-1,self.x+self.width-1] if self.y>0 else 0 
        BL = ii[self.y+self.height-1,self.x-1] if self.x>0 else 0 

        return BR + TL - TR-BL