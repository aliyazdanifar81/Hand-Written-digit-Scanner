import Levenshtein


def cer(true: list, pred: list):
    true, pred = tuple(true), tuple(pred)
    distance = Levenshtein.distance(true, pred)
    cer = (distance / len(true)) * 100
    return cer
