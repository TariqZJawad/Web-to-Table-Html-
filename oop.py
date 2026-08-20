# this is my train code
import re
#print("file container")
#First_name="Vili"
#print(First_name)
#print(f"hello {First_name}")
#age = 22 #int
#score= 5.5 # float
#gpa = 83.34
#is_graduat = True #Boolean
#if is_graduat :
#    print('you are graduate')
#else :
#    print(' you are not graduate yet')
# typeCasting convert of type of data to another
#print(type(gpa))
#ngpa=int(gpa)
#print(ngpa)
#a= b = 1
#a=a+2
#b+=2
#print( a, b)
def q(q):
    print("="*50)
def a(a):
    print(f"----{a}----")
q(q)
#a("input a data")
#b=input("Enter company name: ")
# data from input is str \\
#print(b)
q(q)
a(" practice: calculate Area of Rectangle")
#length=float(input(" what length: "))
#width=float(input(" what width: "))
#Area = length * width
#print(f" the area is {Area} m^2")
a(" Ex2 : shopping cart program")
#item=input(" ehat item would you like to buy ? ")
#price= float(input("what is the price? "))
#quantity=int( input("how many would you like ? "))
#total = price * quantity
#print(f" you have bought {quantity} x {item}/s")
#print(f"your total is {total}")
q(q)
#a('math')
#import math
#print(math.pi)
#print(math.e)
#print(math.sqrt(9))
#print(math.ceil(9.1))
#print(math.floor(9.7))
#q(q)

#redius=float(input('Enter the radius of circle: '))
#c=2*math.pi*redius
#A= math.pi * redius**2
#print(f"the circumference is :{round(c, 2)} and the Area is {round(A,3)}")
q(q)
#a('if')
#age=int(input('how old are youu :'))
#if age < 13 :
#    print('yoy are a child')
#elif age < 18 :
#    print('you are a guy')
#elif age < 20 :
#    print('you are teenage')
#elif age< 50 :
#    print('you are adult')
#else :
#    print('you are old')
#---------------------------------------------------------------
q(q)
a("new course python for Data science Ai and Develope by IBM on Coursera")
print(2 + 15)
print(type(12))
print(type(2.14),
type("h"),
float(2),
int(2.54),
float("2.12"),
str(5),
type(True),
int(True),
str(False),
bool(1),
bool(0),
bool(), end= " \n ")
print(6/2)
type(print(6 // 2))
print(30+20+40)
print((55-5)/10)
print((6*10)/12)
x=3+2*2
print(x)
y=(3+2)*2
print(y)
z= x+y
print(z)
print(11//2)
x=4
x=x/2
print(x)
name= 'The BodyGuard'
print(
    "the BodyGuard",
    'The BodyGuard',
    '1 2 3 4 5 6 ',
    '@#$%^&',
    name[0],
    name[3],
    name[-2],
    len(name),
    name[0:4] +
    name[8:12],
    name[::2],
    name[0:5:2],
    name + " is the best album",
    name *3 ,
    "the BodyGuard\n is the best album",
    "the BodyGuard\t is the best album",
    "the BodyGuard\\ is the best album",
    r"the BodyGuard\n is the best album",
    name.upper,
    name.capitalize,
    name.replace('BodyGuard','Janet'),
    name.find('he'),
    name[(name.find(' ')+1):],
    name.split(),
    end="\n")
print(2+3*4/2)


#List
L= ['The bodyGuard', 7.0, 1992, (5, 'tan'), True]
print('the same element using negative and positive indexing:\n Postive:',L[0],
'\n Negative:' , L[-3]  )
print('the same element using negative and positive indexing:\n Postive:',L[1],
'\n Negative:' , L[-2]  )
print('the same element using negative and positive indexing:\n Postive:',L[2],
'\n Negative:' , L[-1]  )
print(L[3:5])
L.extend(['pop',7.0])
print(L)
L[2]= -0.5
print(L)
L.append(2026)
del(L[2])
print(L)

B=L[:]
L.append([2, 4, 5, 6])
print(L)
print(B)
#Senario
shopping_list = []
shopping_list.append(["watch", "Laptop", "Shoes", "Pen", "Clothes"])
print(shopping_list[0])
print(shopping_list[-1])
print(shopping_list[1:3])
print(shopping_list)
shopping_list[0][3] = "NoteBook"
print(shopping_list)
del(shopping_list[0][-1])
print(shopping_list)
#Tuples
#Dictionaries
Dict= {"key1":1 , "Key2": "2", "key3": [3, 3, 3], "key4": (1, 3, 6, 8, 9)}
print(Dict)
print(Dict["key1"])
release_year_dict = {"Thriller": "1982", "Back in Black": "1980", \
                    "The Dark Side of the Moon": "1973", "The Bodyguard": "1992", \
                    "Bat Out of Hell": "1977", "Their Greatest Hits (1971-1975)": "1976", \
                    "Saturday Night Fever": "1977", "Rumours": "1977"}
print(release_year_dict)
print(release_year_dict['Thriller'])
print(release_year_dict.keys)
print(release_year_dict.values)
release_year_dict['Graduation']= '2007'
print(release_year_dict)
#sets
set1 = {"pop", "rock", "soul", "hard rock", "rock", "R&B", "rock", "disco"}
print(set1)
album_list = [ "Michael Jackson", "Thriller", 1982, "00:42:19", \
              "Pop, Rock, R&B", 46.0, 65, "30-Nov-82", None, 10.0]
album_set = set(album_list)             
print(album_set)
album_set1 = set(["Thriller", 'AC/DC', 'Back in Black'])
album_set2 = set([ "AC/DC", "Back in Black", "The Dark Side of the Moon"])
set3= album_set1 & album_set2
print(set3)
print(album_set1.difference(album_set2))
print(album_set2.difference(album_set1))
print(album_set1.intersection(album_set2))
print(album_set1.union(album_set2))
print(set(album_set1).issubset(album_set2))
print(set(album_set2).issubset(album_set1))
A = [1, 2, 2, 1]  
B = set([1, 2, 2, 1])
print("the sum of A is:", sum(A))
print("the sum of B is:", sum(B))
list=[]
print(list)
range(3)
dates=[1982, 1980, 1973]
N= len(dates)
for i in range (N):
    print(dates[i])
for i in range(0,8):
    print(i)
    list.append(i)
    print(list)
squares = ['red', 'yellow', 'green', 'purple', 'blue']
for i in range(0,5):
    print(" before square ", i, 'is', squares[i])
    squares[i]= 'white'
    print("after square ", i, 'is', squares[i])
s1=['red', 'yellow', 'green', 'purple', 'blue']
for i, s in enumerate(s1):
    print(i, s)
dates.append(2000)
i=0
year=dates[0]
while(year != 1973):
    print(year)
    i+=1
    year=dates[i]
print("It took ", i, 'repetitions to get out of loop ')
for num in range(1, 10):
    if num == 5:
        print("Breaking the loop at:", num)
        break
    print(num)
for num in range(1, 6):
    if num == 3:
        continue
    print(num)
count = 0
while count < 10:
    count += 1
    if count == 3:
        continue  # skip printing 3
    if count == 8:
        break     # stop the loop when count is 8
    print(count)
u= -6
while u<5:
    u +=1
    print(u)
for i in range(-5,6):
    print(i)
PlayListRatings = [10, 9.5, 10, 8, 7.5, 5, 10, 10]
i = 0
Rating = PlayListRatings[0]
while(i < len(PlayListRatings) and Rating >= 6):
    print(Rating)
    i = i + 1 # This prints the value 10 only once 
    Rating = PlayListRatings[i]
    i = i + 1
squares = ['orange', 'orange', 'purple', 'blue ', 'orange']
new_squares = []
i=0
while i < len(squares) and squares[i] == 'Orange' :
    new_squares.append(squares[i])
    i= i+1
print(new_squares)
def calculate_total(a, b):  # Parameters: a and b
    total = a + b           # Task: Addition
    return total            # Output: Sum of a and b

result = calculate_total(5, 7)  # Calling the function with inputs 5 and 7
print(result)  # Output: 12
def MJ():
    print('The BodyGuard')
    
def MJ1():
    print('The BodyGuard')
    return(None)
MJ()
MJ1()
print(MJ())
print(MJ1())
def type_of_album(album, year_released):
    
    print(album, year_released)
    if year_released > 1980:
        return "Modern"
    else:
        return "Oldie"
    
x = type_of_album("The BodyGuard", 1980)
print(x)
def PrintList(the_list):
    for element in the_list:
        print(element)
PrintList(['1', 1, 'the man', "abc"])
# Python Program to Count words in a String using Dictionary
def freq(string):
    
    #step1: A list variable is declared and initialized to an empty list.
    words = []
    
    #step2: Break the string into list of words
    words = string.split() # or string.lower().split()
    
    #step3: Declare a dictionary
    Dict = {}
    
    #step4: Use for loop to iterate words and values to the dictionary
    for key in words:
        Dict[key] = words.count(key)
        
    #step5: Print the dictionary
    print("The Frequency of words is:",Dict)
    
#step6: Call function and pass string in it
freq("Mary had a little lamb Little lamb, little lamb Mary had a little lamb.Its fleece was white as snow And everywhere that Mary went Mary went, Mary went \
Everywhere that Mary went The lamb was sure to go")


# Python Program to Count words in a String using Dictionary
def freq(string,passedkey):

    #step1: A list variable is declared and initialized to an empty list.
    words = []

    #step2: Break the string into list of words
    words = string.split() # or string.lower().split()

    #step3: Declare a dictionary
    Dict = {}

    #step4: Use for loop to iterate words and values to the dictionary
    for key in words:
        if(key == passedkey):
            Dict[key] = words.count(key)   
    #step5: Print the dictionary
    print("Total Count:",Dict)

#step6: Call function and pass string in it
freq("Mary had a little lamb Little lamb, little lamb Mary had a little lamb.Its fleece was white as snow And everywhere that Mary went Mary went, Mary went \
Everywhere that Mary went The lamb was sure to go","little")
a = 1

try:
    b = int(input("Please enter a number to divide a"))
    a = a/b
except ZeroDivisionError:
    print("The number you provided cant divide 1 because it is 0")
except ValueError:
    print("You did not provide a number")
except:
    print("Something went wrong")
else:
    print("success a=",a)
finally:
    print("Processing Complete")

class car:
    max_speed = 120
    def __init__(self, make, model, color, speed = 0):
        self.make = make
        self.model = model
        self.color = color
        self.speed = speed 
    def acceleraton(self, acceleration):
        if self.speed + acceleraton <= car.max_speed :
            self.speed += acceleration
        else:
            self.speed = car.max_speed
    def get_speed(self):
        return self.speed

# Create objects (instances) of the Car class
car1 = Car.make("Toyota", "Camry", "Blue")
car2 = Car.make("Honda", "Civic", "Red")
car1.acceleration(30)
print(f" {car1.make} {car1.model} is currently at {car1.get_speed()}")
