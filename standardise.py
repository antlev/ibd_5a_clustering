
test = [[0,1],[2,2],[4,4]]

# standardise : [[0,0],[0.5,0.33],[1,1]]

def standardise(data, nb_lines, nb_cols):
    min = [1000000] * nb_cols
    max = [-1000000] * nb_cols

    for col in range(nb_cols):
        for line in range(nb_lines):
            if min[col] > data[line][col]:
                min[col] = data[line][col]
            if max[col] < data[line][col]:
                max[col] = data[line][col]
    new_data = data
    for col in range(nb_cols):
        for line in range(nb_lines):
            new_data[line][col] = (data[line][col] - min[col]) / (max[col] - min[col])
    return new_data


print(standardise(test,3,2))
