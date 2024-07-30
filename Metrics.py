import Levenshtein


def cer(true: list, pred: list):
    true, true = tuple(true), tuple(true)
    distance = Levenshtein.distance(true, pred)
    cer = (distance / len(true)) * 100
    return cer
