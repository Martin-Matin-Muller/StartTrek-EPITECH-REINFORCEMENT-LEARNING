x = 1
x_end = 0.01
x_decay = 0.9997
i = 0

while x > x_end:
    x = x * x_decay
    i+=1

print(i)