from utils import *
from sys import argv

attack_file = "../attacks/DOS_ATCK.txt" if len(argv) == 1 else argv[1]
print(attack_file)
file  = open(attack_file, 'r')
text = file.read()

# TODO:
# Show the efect of threshold and tolerance
# Compare Filter with and without time, e.g., DOS and FALSIFYING
# Show how the time works without the check for the previous message being normal(utils line 93)
filter = Filter(CommunicationMatrix('./communication_matrix.json'), threshold=2, tolerance=0.03, enable_time=True)
countFake = 0
countFiltered = 0
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
        # print("Incorrect message:")
        # print(str(msg))
        # exit(0)
    # Analyze the first attack message that was not filtered
    # if msg.label == 'Attack' and filter.test(msg) == 'Normal':
    #     print(str(msg))
    #     exit(0)

print(f'Fake: {countFake} Filtered: {countFiltered} Incorrect: {countIncorrect}')
