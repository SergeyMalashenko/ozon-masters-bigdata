from pyspark import SparkContext, SparkConf
from collections import defaultdict
import logging
import sys

conf = SparkConf()
sc = SparkContext(appName="shortest_path", conf=conf)
logging.basicConfig(level=logging.DEBUG)
logging.info("PySpark script logger initialized")

start_node = int(sys.argv[1])
end_node   = int(sys.argv[2])

input_file  = sys.argv[3]
output_file = sys.argv[4]

raw_graph = sc.textFile(input_file)
graph_data = raw_graph.map(lambda x : x.split("\t")).map(lambda x : (int(x[1]), int(x[0]))).cache()

ITERATIONS = 1000
front_node_s, used_node_s = set(), set()

front_node_s.add(start_node)
parent_s = defaultdict(list)
for i in range(0, ITERATIONS):
    front_sc, used_sc = sc.broadcast(front_node), sc.broadcast(used_node)
    if (end_node in used_node_s):
        break
    output = graph.filter(lambda x : (x[0] in front_sc.value) and (x[0] not in used_sc.value)).groupByKey().mapValues(list).collect()
    
    used_node_s.update(front_node)
    front_node = set()
    for val in output:
        for item in val[1]:
            parent_s[item].append(val[0])
        front_node_s.update(val[1])
path_length = i

def get_parents(current_node, target_node, parents, level, max_level):
    if level >= max_level:
        return []
    if current_node == target_node:
        return [[target_node]]
    ret_val = list()
    for parent in parents[current_node]:
        temp_val = []
        for prev in get_parents(parent,target_node, parents, level+1, max_level):
            temp_val.append(current_node)
            temp_val.extend(prev)
        if (len(temp_val) != 0):
            ret_val.append(temp_val)
    return ret_val

out = get_parents(end_node, start_node, parent_s, 0, path_length)

answer = list()
for v in out:
    path = list(v)
    path.reverse()
    answer.append(','.join(map(lambda x :str(x), path)))

ready_answer = "\n".join(answer)
answer_rdd = sc.parallelize([ready_answer])
answer_rdd.saveAsTextFile(output_file)

sc.stop()
