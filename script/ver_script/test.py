a = {'a': 9, 'b': 1, 'c': 5, 'd': 6, 'e': 2}

L = sorted(a.items(), key=lambda item: item[1], reverse=True)

L = L[:3]
test = L[1]
id = test[0]
value = test[1]
print(L)
