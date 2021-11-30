import os

for i in range(7, 12):
    for samples in range(1, 10):
        width = 2 ** i
        sample_count = samples * 8
        os.system(f"python main_bitmasked.py {width} {sample_count}")

