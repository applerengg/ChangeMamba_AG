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
    names = ["Baseline", "AGBD", "AGB", "ALIGN", "FOCAL", "FOCAL + ALIGN", "FOCAL + AGB", "ALIGN + AGB", "FOCAL + ALIGN + AGBD", "FOCAL + ALIGN + AGB"]
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
"lofF1 is 64.9321, clfF1 is 0.0000, oaF1 is 19.4796, sub class F1 score is [96.9521  0.5687  0.      4.4666]"						
"lofF1 is 55.8843, clfF1 is 0.0000, oaF1 is 16.7653, sub class F1 score is [96.9946  0.11    0.      2.2794]"						
"lofF1 is 54.7425, clfF1 is 0.2372, oaF1 is 16.5888, sub class F1 score is [9.21178e+01 1.43250e+00 6.24000e-02 7.75480e+00]"						
"lofF1 is 52.8744, clfF1 is 0.0000, oaF1 is 15.8623, sub class F1 score is [97.2553  0.7711  0.      1.955 ]"						
"lofF1 is 62.2707, clfF1 is 1.1755, oaF1 is 19.5041, sub class F1 score is [94.8592  1.6874  0.3774  6.6679]"						
"lofF1 is 64.1689, clfF1 is 0.4554, oaF1 is 19.5695, sub class F1 score is [92.2075  0.2036  0.284   2.9412]"						
"lofF1 is 64.5192, clfF1 is 0.0000, oaF1 is 19.3558, sub class F1 score is [96.9371  0.6195  0.      0.7896]"						
"lofF1 is 61.0427, clfF1 is 1.1683, oaF1 is 19.1306, sub class F1 score is [96.4557  0.6641  0.7976  1.5293]"						
"lofF1 is 64.1302, clfF1 is 1.8919, oaF1 is 20.5634, sub class F1 score is [94.7007  1.9642  0.7356  4.2513]"						
"lofF1 is 63.6664, clfF1 is 1.6225, oaF1 is 20.2357, sub class F1 score is [93.7038  1.6699  0.5976  5.4801]"						
"""
    multiple(s)

if __name__ == "__main__":
    main()
