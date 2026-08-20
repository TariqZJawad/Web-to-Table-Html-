import re
match= re.search(r"\d\d\d\d\d\d\d\d\d\d", " My Phone number is 1234567890")
if match:
    print("Phone number found: ", match.group())
else:
    print("No match")

pattern = r"\W"
text = "Hello, world!"
matches = re.findall(pattern, text)
print("Matches: ", matches)

s2= " The bodyGuard is  the best album of 'Whitney Houstos'."
result= re.findall("st", s2)
print(result)
split_array = re.split(r"\s", s2)
print(split_array)
#replace
pattern = r"Witney Houston"
replacement = "legend"
new_tring= re.sub(pattern, replacement, s2 )
print(new_tring)
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
#################################
##### Text Analysis Practice ####
givenstring="Lorem ipsum dolor! diam amet, consetetur Lorem magna. sed diam nonumy eirmod tempor. diam et labore? et diam magna. et diam amet."
class TextAnalyzer(object):
    def __init__(self, text):
        # remove punctuation
        formattedText = text.replace(',', '').replace('.','').replace('!','').replace('?','')
        #make text lowercase
        formattedText = formattedText.lower()
        self.fmtText = formattedText
    def freqall(self):
        #split text into words
        wordList= self.fmtText.split(' ')
        #create dictionary
        freqMap = {}
        for word in set(wordList) :
            freqMap[word] = wordList.count(word)
        return freqMap
    def freqOf(self,word):
        freqDict = self.freqall()
        if word in freqDict:
            return freqDict[word]
        else:
            return 0
        
analyzed = TextAnalyzer(givenstring)        
print("Formatted Text:", analyzed.fmtText)
freqMap = analyzed.freqall()
print(freqMap)
word = "lorem"
frequency = analyzed.freqOf(word)
print("The word",word,"appears",frequency,"times.")