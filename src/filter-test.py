from utils import Filter, CommunicationMatrix, CANMsgFromline, get_cmd_args
from sys import argv

attack_file = "../attacks/DOS_ATCK.txt" if len(argv) == 1 else argv[1]
print(attack_file)
file  = open(attack_file, 'r')
text = file.read()

argv.pop(1)
opt = get_cmd_args()

# TODO:
# Show the efect of threshold and tolerance
# Compare Filter with and without time, e.g., DOS and FALSIFYING
# Show how the time works without the check for the previous message being normal(utils line 93)
filter = Filter(CommunicationMatrix('./communication_matrix.json'),
                window_size = opt.window_size,
                stride = opt.stride,
                threshold=2,
                tolerance=4e-2,
                enable_time=False)
cnt = 0
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
    prev_window = filter.current_window
    win = prev_window + [msg]
    real_label = 'Attack' if any(m.label != 'Normal' for m in win) else 'Normal'
    res = filter.test(msg)
    if res is None:
        continue
    elif res == 'Attack' and real_label == 'Attack':
        true_positive += 1
    elif res == 'Normal' and real_label == 'Normal':
        true_negative += 1
    elif res == 'Attack' and real_label == 'Normal':
        false_negative += 1
        for m in prev_window:
            print(str(m))
        filter.current_window = prev_window[:-1]
        filter.test(msg, debug=True)
        exit(0)
    elif res == 'Normal' and real_label == 'Attack':
        false_positive += 1
    else:
        print(res)
        print(real_label)
        print('Unknown label')
        exit(1)
    cnt += 1

print(f"True Positive: {true_positive/cnt}")
print(f"True Negative: {true_negative/cnt}")
print(f"False Positive: {false_positive/cnt}")
print(f"False Negative: {false_negative/cnt}")
