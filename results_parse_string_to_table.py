import re

def multiple(s: str):
    print(s)
    s = s.strip().replace('\"', '')
    results = []
    for line in s.split('\n'):
        out = single(line)
        results.append(out)
    
    print()
    print("=" * 80)
    names = ["Baseline", 
             "AGBD", "AGB", "ALIGN", "FOCAL", 
             "FOCAL + ALIGN", "FOCAL + AGB", "ALIGN + AGB", 
             "FOCAL + ALIGN + AGBD", "FOCAL + ALIGN + AGB"]
    for name, res in zip(names, results):
        print(f"{name:<21} & {res}")
        if name in ("Baseline", "FOCAL"):
            print(r"\hline")
    return results


def single(s: str):
    # s = input()
    """lofF1 is 60.2746, clfF1 is 16.4003, oaF1 is 29.5626, 
    sub class F1 score is [91.2971  5.4221 41.4387 41.0129]"""
    # Parse the input string
    
    loff1_match = re.search(r'lofF1 is ([\d.]+(?:e[+-]?\d+)?)', s)
    clff1_match = re.search(r'clfF1 is ([\d.]+(?:e[+-]?\d+)?)', s)
    oaf1_match = re.search(r'oaF1 is ([\d.]+(?:e[+-]?\d+)?)', s)
    classes_match = re.search(r'sub class F1 score is \[([\d.\se+-]+)\]', s)
    
    # Extract values
    loff1 = float(loff1_match.group(1)) if loff1_match else ""
    clff1 = float(clff1_match.group(1)) if clff1_match else ""
    oaf1  = float(oaf1_match.group(1))  if oaf1_match  else ""
    
    # Extract and split class scores
    classes = []
    if classes_match:
        classes = classes_match.group(1).split()
    
    # Format output
    result = f"{loff1:>5.2f} & {clff1:>5.2f} & {oaf1:>5.2f}"
    for cls in classes:
        result += f" & {float(cls):>5.2f}"

    result += r" \\"

    print(result)
    return result


def main():
    s = """
lofF1 is 90.6886, clfF1 is 29.9680, oaF1 is 48.1842, sub class F1 score is [98.4292 20.1287 17.057  66.6268]
lofF1 is 90.7547, clfF1 is 34.9965, oaF1 is 51.7239, sub class F1 score is [98.9553 22.4364 22.0943 69.6343]
lofF1 is 91.3124, clfF1 is 36.8642, oaF1 is 53.1986, sub class F1 score is [98.9357 27.1406 21.2716 68.7638]
lofF1 is 91.2827, clfF1 is 29.0642, oaF1 is 47.7298, sub class F1 score is [98.9792 15.7504 20.0522 70.6072]
lofF1 is 91.1897, clfF1 is 30.3317, oaF1 is 48.5891, sub class F1 score is [98.9229 20.2922 17.1651 70.282 ]
lofF1 is 91.2172, clfF1 is 29.5085, oaF1 is 48.0211, sub class F1 score is [98.9156 15.1433 22.2878 68.7689]
lofF1 is 90.6949, clfF1 is 35.1392, oaF1 is 51.8059, sub class F1 score is [98.7216 33.0546 16.762  72.5078]
lofF1 is 91.2355, clfF1 is 27.5336, oaF1 is 46.6442, sub class F1 score is [98.9237 20.1507 13.714  79.2146]
lofF1 is 90.8690, clfF1 is 36.1999, oaF1 is 52.6006, sub class F1 score is [98.4562 30.5143 20.1857 55.4657]
lofF1 is 91.1571, clfF1 is 28.8498, oaF1 is 47.5420, sub class F1 score is [97.785  17.8403 17.6037 64.2534]
"""
    multiple(s)

if __name__ == "__main__":
    main()
