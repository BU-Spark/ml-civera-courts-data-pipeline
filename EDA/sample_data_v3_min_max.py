import json
import os
import random
import time

from tqdm import tqdm

from utils import *

total_records = 73173625
table_name = "cdocs_case_action_index"


os.makedirs("sample", exist_ok=True)

def get_sampled_data(_limit=1000):
    """

    :param _limit:
    :return: dict: {action_type: {case_action_id, actor, action, description}}
    """

    with open("count/count.json", "r") as f:
        action_counts = json.load(f)["action_counts"]

    results = {}

    connection = get_connection()

    import time

    try:
        with connection.cursor() as cursor:
            # 查询case_action_id的最大值和最小值
            cursor.execute(f"SELECT MIN(case_action_id) AS min_id, MAX(case_action_id) AS max_id FROM {table_name}")
            bounds = cursor.fetchone()
            min_id, max_id = bounds['min_id'], bounds['max_id']

            # 定义步进值
            # step = (max_id - min_id) // 100  # 例如，我们按照范围的1%进行步进
            count = 10000
            step = (max_id - min_id) // count
            print(step)

            # 初始化列表来保存每个采样点的数据
            samples = []

            t = time.time()
            success = 0
            # 使用for循环按步进值进行采样并执行点查询
            for _id, sample_id in enumerate(tq := tqdm(range(min_id, max_id, step))):
                tq.set_description(f"{success}/{_id}")
                cursor.execute(f"""
                    SELECT case_action_id, actor, action, description 
                    FROM {table_name} 
                    WHERE case_action_id = {sample_id}
                """)
                result = cursor.fetchone()  # 执行查询并获取结果
                if result:
                    success += 1
                    samples.append(result)  # 如果查询有结果，则添加到samples列表中

                    # if success % 100 == 0:
                    #     print("Avg time per sample: {} s".format(round((time.time() - t) / len(samples), 4)))

            # 输出采样点的数据
            # for sample in samples:
            print(len(samples), success)
            print("Avg time per sample:", round((time.time() - t) / len(samples), 4))

    except Exception as e:
        print(e)

    finally:
        connection.close()

    return results


limit = 100
data = get_sampled_data(limit)


exit()


with open(f"sample/sampled_data_based_on_action_type_limit={limit}.json", "w") as f:
    json.dump(data, f)

print("Transforming... \n"
      "{action_type: [{case_action_id, actor, action, description}]} -> {case_action_id: {actor, action, description}}")
# {action_type: {case_action_id, actor, action, description}} -> case_action_id: {actor, action, description}
data_list = {}
for action_type, rows in tqdm(data.items()):
    for row in rows:
        action_case_id = row.pop("case_action_id")  # remove 'action_case_id'
        data_list[action_case_id] = row

dump_json(data_list, f"sample/sampled_data_limit={limit}.json")
