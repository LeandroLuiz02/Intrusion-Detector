from utils import *
from sys import argv

attack_file = "../attacks/DOS_ATCK.txt" if argv[1] == "" else argv[1]
print(attack_file)
file  = open(attack_file, 'r')
text = file.read()

filter = Filter(CommunicationMatrix('./communication_matrix.json'), threshold=2)
countFake = 0
countFiltered = 0

for line in text.split('\n'):
    msg = CANMsgFromline(line)
    if msg is None: break
    if msg.label == 'Attack':
        countFake += 1
        if filter.test(msg) == 'Attack':
            countFiltered += 1
    if filter.test(msg) == 'Attack' and msg.label == 'Normal':
        # The Filter must never classify a normal message as an attack
        print("Incorrect message:")
        print(str(msg))
        exit(0)
    # Analyze the first attack message that was not filtered
    # if msg.label == 'Attack' and filter.test(msg) == 'Normal':
    #     print(str(msg))
    #     exit(0)

print(f'Fake: {countFake} Filtered: {countFiltered} Incorrect: {countIncorrect}')
