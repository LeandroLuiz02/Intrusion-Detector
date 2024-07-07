from utils import *

file  = open("../attacks/DOS_ATCK.txt", 'r')
text = file.read()

filter = Filter(CommunicationMatrix('./communication_matrix.json'), threshold=2)
countFake = 0
countFiltered = 0
# The Filter must never classify a normal message as an attack
countIncorrect = 0

for line in text.split('\n'):
    msg = CANMsgFromline(line)
    if msg is None: break
    if msg.label == 'Attack':
        countFake += 1
        if filter.test(msg) == 'Attack':
            countFiltered += 1
    if filter.test(msg) == 'Attack' and msg.label == 'Normal':
        countIncorrect += 1

print(f'Fake: {countFake} Filtered: {countFiltered} Incorrect: {countIncorrect}')
