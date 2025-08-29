# Image-Quality-Assessment-Made-Practical-for-Smartphone-Photography

[![Academic Use Only](https://img.shields.io/badge/License-Academic%20Only-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)


> A practical No-Reference Image Quality Assessment (NR-IQA) method specifically designed for modern smartphone photography challenges. Trained on real-world smartphone images with subjective labels.

## Introduction

We establish the relationships among quality labels by constructing a HEX graph informed by prior knowledge. Based on this framework, we propose a probabilistic multi-label ordinal regression model formulated as a conditional random field (CRF).

## Project Progress Tracker

### ✅ Completed Milestones

- Constructed a 6-label HEX graph for initial validation

```
Labels = [ 整体偏亮, 整体偏暗, 整体对比度高, 整体对比度低, 高光过曝, 高光压制过度]
Levels = { "1": "不存在相应问题", "2": "普通", "3": "严重", "4": "阻塞"}
Nodes = [
    {"label":"整体偏亮","levels":[1,2,3,4]},
    {"label":"整体偏暗","levels":[1,2,3,4]},
    {"label":"整体对比度高","levels":[1,2,3,4]},
    {"label":"整体对比度低","levels":[1,2,3,4]},
    {"label":"高光过曝","levels":[1,2,3,4]},
    {"label":"高光压制过度","levels":[1,2,3,4]}
  ]
Exclusions = [
		# undirected edges
    {"label_a":"整体偏亮","a_levels":[2,3,4], "label_b":"整体偏暗","b_levels":[2,3,4]},
    {"label_a":"整体对比度高","a_levels":[2,3,4], "label_b":"整体对比度低","b_levels":[2,3,4]},
    {"label_a":"高光过曝","a_levels":[2,3,4], "label_b":"高光压制过度","b_levels":[2,3,4]}
Subsumptions = [
		# directed edges
    {"label_a":"整体偏亮", "label_b":"整体偏暗", "map":[[2,1]]},
    {"label_a":"整体偏亮"，"label_b":"整体偏暗", "map":[[3,1]]},
    {"label_a":"整体偏亮"，"label_b":"整体偏暗", "map":[[4,1]]},
    {"label_a":"整体偏暗"，"label_b":"整体偏亮", "map":[[2,1]]},
    {"label_a":"整体偏暗"，"label_b":"整体偏亮", "map":[[3,1]]},
    {"label_a":"整体偏暗"，"label_b":"整体偏亮", "map":[[4,1]]},
    {"label_a":"整体对比度高"，"label_b":"整体对比度低", "map":[[2,1]]},
    {"label_a":"整体对比度高"，"label_b":"整体对比度低", "map":[[3,1]]},
    {"label_a":"整体对比度高"，"label_b":"整体对比度低", "map":[[4,1]]},
    {"label_a":"整体对比度低"，"label_b":"整体对比度高", "map":[[2,1]]},
    {"label_a":"整体对比度低"，"label_b":"整体对比度高", "map":[[3,1]]},
    {"label_a":"整体对比度低"，"label_b":"整体对比度高", "map":[[4,1]]},
    {"label_a":"高光过曝"，"label_b":"高光压制过度", "map":[[2,1]]},
    {"label_a":"高光过曝"，"label_b":"高光压制过度", "map":[[3,1]]},
    {"label_a":"高光过曝"，"label_b":"高光压制过度", "map":[[4,1]]},
    {"label_a":"高光压制过度"，"label_b":"高光过曝", "map":[[2,1]]},
    {"label_a":"高光压制过度"，"label_b":"高光过曝", "map":[[3,1]]},
    {"label_a":"高光压制过度"，"label_b":"高光过曝", "map":[[4,1]]}
]
```
  
- Translated MATLAB-based HEX graph implementation into our framework
  


### 🔄 In Progress

### 🔜 Next Steps
