

with open('/root/autodl-tmp/data/Track1/train.txt') as f:
    lines = f.readlines()
point_max_length = 0
sentence_max_length = 0
for line in lines:
    point, sentence, label = line.split('\t')
    point_max_length = point_max_length if point_max_length > len(point.split(' ')) else len(point.split(' '))
    sentence_max_length = sentence_max_length if sentence_max_length > len(sentence.split(' ')) else len(sentence.split(' '))
print(point_max_length, ' ', sentence_max_length)
print(len(lines))