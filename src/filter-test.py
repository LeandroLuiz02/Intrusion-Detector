from utils import Filter, CommunicationMatrix, CANMsgFromline
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
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

msgs = text.split('\n')
for line in msgs:
    msg = CANMsgFromline(line)
    if msg is None: break
    if msg.label is None:
        print('Unknown label')
        continue
    res = filter.test(msg)
    if res == 'Attack' and msg.label == 'Attack':
        true_positive += 1
    elif res == 'Normal' and msg.label == 'Normal':
        true_negative += 1
    elif res == 'Attack' and msg.label == 'Normal':
        false_negative += 1
    elif res == 'Normal' and msg.label == 'Attack':
        false_positive += 1
    else:
        print(res)
        print(msg.label)
        print('Unknown label')
        exit(1)

print(f"True Positive: {true_positive/len(msgs)}")
print(f"True Negative: {true_negative/len(msgs)}")
print(f"False Positive: {false_positive/len(msgs)}")
print(f"False Negative: {false_negative/len(msgs)}")
