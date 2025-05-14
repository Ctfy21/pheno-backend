import pandas as pd
import numpy as np
import json


def excel_to_plantArray(df):
    try:
        array_head = df.head(0).columns.tolist()
        temp_array = []
        for index, value in enumerate(array_head):
            if not "Unnamed" in value:
                temp_array.append([index, value, 0])

        temp_array.append([len(array_head), "total"])

        for index, value in enumerate(temp_array[0:-1]):
            temp_array[index][2] = temp_array[index + 1][0] - temp_array[index][0]

        temp_array.pop()
        final_array = []
        plant_array = []

        for value in df.values:
            for row in temp_array:
                x = value[row[0]:row[0]+row[2]]
                x = x[~pd.isnull(x)]
                if len(x) == 1:
                    plant_array.append([row[1], x[0].item() if isinstance(x[0], np.generic) else x[0]])
                elif len(x) == 0:
                    plant_array.append([row[1], None])
                else:
                    plant_array.append([row[1], x.tolist() if isinstance(x, np.ndarray) else x])
            final_array.append(plant_array)
            plant_array = []

        return final_array
    except Exception as e:
        return ["error", e]


# df = pd.read_excel("scripts/test.xlsx", sheet_name="сорт. данные")

# print(excel_to_plantArray(df)[0])
# val = json.dumps(excel_to_plantArray(df)[0][4][1])
# print(type(val))
# print(val)









