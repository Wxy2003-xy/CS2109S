import datetime
import requests
from typing import Callable, Iterable, List, TypeVar
from itertools import chain

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Output type

def flat_map(func: Callable[[T], Iterable[R]], collection: Iterable[T]) -> List[R]:
    return list(chain.from_iterable(map(func, collection)))
date = datetime.date.today()
url = f"https://www.nytimes.com/svc/wordle/v2/{date:%Y-%m-%d}.json"
response = requests.get(url).json()
print(f"Answer: {response['solution']}")

# all n letter word list
def helper(c):
    res = [c] * 26
    for i in range(26):
        res[i] += chr(i + 97)  # Append the corresponding letter (a-z)
    return res

def gen_list(lst, length):
    if length == 0:
        return lst
    tmp = [item for c in lst for item in helper(c)]  # Flatten the result
    return gen_list(tmp, length - 1)

# Write to data file
with open("five_letter.txt", "w") as file: 
    for w in gen_list([""], 5):
        file.write(w + "\n")