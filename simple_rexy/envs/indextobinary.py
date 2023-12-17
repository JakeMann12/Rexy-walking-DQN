def index_to_action(index):
    # Convert the index to a 6-bit binary string
    binary_str = format(index, '06b')
    # Translate the binary string to actions for each joint
    actions = [-2 if bit == '0' else 2 for bit in binary_str]
    return actions


print(index_to_action(0))
print(index_to_action(1))
print(index_to_action(63))
print(index_to_action(64))
print(index_to_action(65))

print(format(0, '06b'))
print(format(1, '06b'))
print(format(2, '06b'))
print(format(7, '06b'))
print(format(8, '06b'))
print(format(63, '06b'))
print(format(64, '06b'))