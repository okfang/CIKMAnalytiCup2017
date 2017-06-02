import numpy as np
import pandas as pd

test_path = "F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\results\\gbdt_result_update2.csv"

test_data = pd.read_csv(test_path,header=None)
sum = test_data[test_data<0]
print(sum)

# print(test_data)
# pd.DataFrame(test_data).to_csv("F:\\data\\tianchi\\CIKM2017\\CIKM2017_train\\data_new\\CIKM2017_train\\results\\gbdt_result_update2.csv",header=False,index=False)