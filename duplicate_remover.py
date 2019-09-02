import os
from hashlib import md5
import time

starttime = time.time()

# generate hash value for file
def file_hash(filepath):
    """
    Function that returns a hash of a file

    Parameters:
    filepath: string of a file path

    Returns:
    string: file hash
    """
    with open(filepath, 'rb') as f:
        return md5(f.read()).hexdigest()


# delete all duplicates files
def remove_duplicates(path):
    """
    Function that loops through the images in a folder
    and remove duplicates
    """

    duplicates = []
    hash_keys = {}

    files_list = os.listdir(path)

    for index, filename in enumerate(files_list):
        filepath=path+"\\"+filename
        if os.path.isfile(filepath):
            filehash = file_hash(filepath)
        if filehash not in hash_keys.keys():  # a novel image
            hash_keys[filehash] = index
        else:  # a duplicate image
            duplicates.append((index, hash_keys[filehash]))

    # remove duplicate images by their paths
    for index, _ in duplicates:
        filepath = path + "\\" + files_list[index]
        os.remove(filepath)

path = r"E:\mixed\Flinders Street Station"
remove_duplicates(path)

endtime = time.time()
print('\nTotal time: %f seconds.' % (endtime - starttime))
