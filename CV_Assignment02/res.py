from numpy.ma.core import maximum

f = open('./res', 'r')

f.readline()

dic = {}
key = 0
threshold = 0
for line in f:
    if 'alpha' in line:
        dic[line[7:-1]] = []
        key = line[7:-1]
    elif 'result' in line:
        txt, threshold, f1 = line.split()
        dic[key].append([int(threshold), float(f1)])

maximum_alpha, maximum_threshold, maximum_f1 = 0, 0, 0
error = []
for key, item in dic.items():
    for lst in dic[key]:
        if maximum_f1 < lst[1]:
            maximum_alpha = key
            maximum_threshold = lst[0]
            maximum_f1 = lst[1]

        elif maximum_f1 == lst[1]:
            error.append([key, lst[0], lst[1]])

if error:
    print("error occured on the following:")
    for e in error:
        print(f"alpha: {e[0]} | threshold: {e[1]} | f1: {e[2]}")

print("\nresult")
print(f"alpha: {maximum_alpha} | threshold: {maximum_threshold} | f1: {maximum_f1}")