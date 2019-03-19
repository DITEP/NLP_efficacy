dic = {}
f = open('types.txt', 'r', encoding='utf-8')
types = []
for line in f:
    line = line.strip()
    types.append(line)
f.close()

f = open('vectors.txt', 'r')
vectors = []
for line in f:
    line = line.strip()
    vectors.append(line)
f.close()

for i, word in enumerate(types):
    dic[word] = vectors[i]

print("Types and vectors length : {} {}".format(len(types), len(vectors)))
with open('vectors_clean.txt', 'w', encoding='utf-8') as fileout:
    fileout.write("{} {}\n".format(len(types), 200))
    for word in dic:
        fileout.write("{} {}\n".format(word, dic[word]))

print("Done")

