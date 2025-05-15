from typing import Union
from math_evaluation import is_equiv
from generation.math_verify.math_equal import math_equal
from generation.math_verify.checker import check_one_answer
from generation.math_verify.util import strip_string, choice_answer_clean

INVALID_ANS = "[invalid]"


def remove_text_box(text):
    if text is None:
        return None
    start = text.find(r"\text{")
    if start == -1:
        return text
    end = None
    stack = []
    answer = text[start:]
    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            start_text = stack.pop()
            if len(stack) == 0:
                end_text = i
                break
    in_text_string = text[start + start_text + 1 : start + end_text]

    if in_text_string.strip() == "and":
        ex_text = text[:start] + text[start + end_text + 1 :]
    else:
        ex_text = (
            text[:start]
            + text[start + start_text + 1 : start + end_text].strip()
            + text[start + end_text + 1 :]
        )
    return ex_text.strip()


def extract_boxed_answer(text, debug=False):
    if text is None:
        return None
    start = text.rfind(r"\boxed{")
    if start == -1:
        return text
    end = None
    stack = []
    answer = text[start:]
    for i, c in enumerate(answer):
        if c == "{":
            stack.append(i)
        elif c == "}":
            start = stack.pop()  # \boxed start{
            if len(stack) == 0:
                end = i  # \boxed end}
                break
    if end is None and debug:
        print("brack not closing", answer)
        return None
    return answer[start + 1 : end]


def extract_math_answer(answer):
    try:
        assert "boxed" in answer
        ans = answer
        extract_ans_temp = ans
        #extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        extract_ans = remove_text_box(extract_boxed_answer(extract_ans))
    except Exception as e:
        # print(e)
        extract_ans = INVALID_ANS
    return extract_ans


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def rstar_equiv(gt, pred):
    # In this function, I integrated multiple open-source evaluation tools
    # each with its own judgment logic and strengths in handling special cases such as LaTeX, units, etc.
    gt = str(gt)
    pred = str(pred)
    try:
        if gt == pred:
            return True

        # For college-math and omni-math, the pred and gt positions need to be changed.
        # Because we found that the quality of ground truth in a small subset of problems within benchmarks like college-math is relatively low.
        if any(
            func(x, y) for func in [math_equal, is_equiv, check_one_answer] for x, y in [(gt, pred), (pred, gt)]
        ):
            return True
        # special for college-math, etc.
        gt_strip, pred_strip = strip_string(gt), strip_string(pred)
        if any(
            func(x, y) for func in [math_equal, is_equiv, check_one_answer] for x, y in [(gt_strip, pred_strip), (pred_strip, gt_strip)]
        ):
            return True

        # for choice question
        if gt in ["A", "B", "C", "D", "E"] and pred not in ["A","B","C","D","E"]:
            pred = choice_answer_clean(pred)
            if math_equal(gt, pred):
                return True
        elif is_multi_choice(gt) and not is_multi_choice(pred):
            pred = "".join(
                    [c for c in pred if c in ["A", "B", "C", "D", "E"]]
                )
            if math_equal(gt, pred):
                return True
    except Exception as e:
        print("maroi_equiv error")
        print(e)
        pass
    return False


def math_equiv(grt: Union[str, list[str]], prd: str):
    prd = (prd)
    if isinstance(grt, list):
        for g in grt:
            if rstar_equiv(g, prd):
                return True
        return False
    else:
        return rstar_equiv(grt, prd)
