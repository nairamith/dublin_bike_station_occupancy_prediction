import math

 
def func(s):
    output = []
    l = math.floor(len(s)/2)
    for i in range(1, l+1):
        temp = "0"*i + "1"*i
        if temp in s:
            output.append(temp)
        temp = "1"*i + "0"*i
        if temp in s:
            output.append(temp)

    return len(output)

print(func(sample))



