def binary_mask(matrix, *maskvalues):
    maskvalues = set(maskvalues)

    mask = [
        [
            0 if index in maskvalues else 1
            for index in row
        ]
        for row in matrix
    ]

    return mask
