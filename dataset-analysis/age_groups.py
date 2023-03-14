'''
Age Groups are defined based on study of Horng et al.
Horng, Wen-Bing, Cheng-Ping Lee, and Chun-Wen Chen. ‘Classification of Age Groups Based on Facial Features’. Tamkang Journal of Science and Engineering 4 (1 September 2001): 183–92.
'''
class AgeGroups:
    ranges = [range(0,3), range(3,13), range(13, 20), range(20, 30),
         range(30, 40), range(40, 50), range(50, 60), range(60, 70),
         range(70, 80), range(80, 90), range(90, 100), range(100,117)]
    
    def getRanges(self):
        return self.ranges
    
    def getMiddles(self):
        middles = []
        for r in self.ranges:
            middles.append(int(round(min(r)+(max(r)-min(r))/2, 0)))
        return middles
    
    