import math

for i in range(2, 11):
    for j in range(1, math.ceil(i/2)):
        P = str(j) + "/" + str(i)
        N = str(i - j) + "/" + str(i)
        p = j / i
        n = (i - j) / i
 
        I = -(p / (p + n)) * math.log2((p / (p + n))) - (n / (p + n)) * math.log2((n / (p + n)))
        print("I(", P, ", ", N, ") = ", round(I, 5))
 