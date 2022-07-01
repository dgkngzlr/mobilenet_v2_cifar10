import numpy as np
import time


def make_gray_code(n: int):

    if n <= 0:
        return []
 
    # will store all generated codes
    arr = list()
 
    # start with one-bit pattern
    arr.append('0')
    arr.append('1')
 
    # Every iteration of this loop generates
    # 2*i codes from previously generated i codes.
    i = 2
    j = 0
    while True:
 
        if i >= 1 << n:
            break
     
        # Enter the previously generated codes
        # again in arr[] in reverse order.
        # Nor arr[] has double number of codes.
        for j in range(i - 1, -1, -1):
            arr.append(arr[j])
 
        # append 0 to the first half
        for j in range(i):
            arr[j] = '0' + arr[j]
 
        # append 1 to the second half
        for j in range(i, 2 * i):
            arr[j] = '1' + arr[j]
            
        i = i << 1
 
    return arr


def make_bit_reversal(l: list):

    for i, _bin in enumerate(l):

        # bit reversal operation
        temp = '0b'+_bin[::-1]

        # convert bit sequence to unsigned int
        l[i] = eval(temp)
    
    return l


def get_hadamard_matrix(k):
    """
    :param k: It is natural number 0 to N
    :return: Hadamard matrix which is size 2^k x 2^k
    """
    if k < 1:
        return np.array([1], dtype=np.float32)

    reformer_matrix = np.ones((2 ** k, 2 ** k), dtype=np.float32)
    reformer_matrix[2 ** (k - 1):, 2 ** (k - 1):] *= -1

    prev_hadamard = get_hadamard_matrix(k - 1)

    # Wk = [[A , B],
    #      [C , D]]
    hadamard_matrix = np.empty((2 ** k, 2 ** k), dtype=np.float32)
    hadamard_matrix[0:2 ** (k - 1), 0:2 ** (k - 1)] = reformer_matrix[0:2 ** (k - 1),
                                                      0:2 ** (k - 1)] * prev_hadamard  # A part
    hadamard_matrix[2 ** (k - 1):, 0:2 ** (k - 1)] = reformer_matrix[2 ** (k - 1):,
                                                     0:2 ** (k - 1)] * prev_hadamard  # B part
    hadamard_matrix[0:2 ** (k - 1), 2 ** (k - 1):] = reformer_matrix[0:2 ** (k - 1),
                                                     2 ** (k - 1):] * prev_hadamard  # C part
    hadamard_matrix[2 ** (k - 1):, 2 ** (k - 1):] = reformer_matrix[2 ** (k - 1):,
                                                    2 ** (k - 1):] * prev_hadamard  # D part

    return hadamard_matrix


def get_walsh_matrix(k):
    """
    :param k: It is natural number 0 to N
    :return: Walsh matrix which is size 2^k x 2^k
    """

    hadamard_matrix = get_hadamard_matrix(k)
    gray = make_gray_code(k)
    reversal = make_bit_reversal(gray)

    # permutation operation
    walsh_matrix = hadamard_matrix[reversal, :]

    return walsh_matrix

def fwht(matrix):

    col = matrix.shape[1]

    walsh = get_walsh_matrix(int(np.log2(col)))
    return (1 / col) * np.matmul(walsh,matrix)

def ifwht(matrix):

    col = matrix.shape[1]

    walsh = get_walsh_matrix(int(np.log2(col)))
    return np.matmul(walsh, matrix)
if __name__ == "__main__":

    n = 14 # 2^n x 2^n matrix



    begin = time.time()
    a = get_hadamard_matrix(n)
    print("Elapsed Time(s): ", time.time() - begin)

    time.sleep(1)

    begin = time.time()
    a = get_walsh_matrix(n)
    print("Elapsed Time(s): ", time.time() - begin)
