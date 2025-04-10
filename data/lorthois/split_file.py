input_file = 'Network1_Raw_Data.txt'
output_prefix = 'Network1_Split_'

with open(input_file, 'r') as file:
    content = file.read()

sections = content.split('\n@')
for i, section in enumerate(sections):
    with open(f'{output_prefix}{i}.txt', 'w') as output_file:
        if i == 0:
            output_file.write(section)
        else:
            output_file.write('@' + section)
