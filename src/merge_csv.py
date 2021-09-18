import pandas as pd
import os 
PATH = "/home/prl/tycho_ws/src/tycho_calibration"

if __name__ == "__main__":
    p1 = os.path.join(PATH, "fk0916_static.csv")
    p2 = os.path.join(PATH, "fk0916_dynamic.csv")
    A = pd.read_csv(p1)
    B = pd.read_csv(p2)
    print(A.shape)
    print(B.shape)
    merged = pd.concat([A, B], axis=0, ignore_index=True)
    print(merged.shape)
    merged.to_csv("output.csv", index=False)
