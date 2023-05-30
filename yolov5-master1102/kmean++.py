from __future__ import division, print_function

import numpy as np
import random
import math


def iou(box, clusters):
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_

def iou_kpp(box, clusters):
    x = np.minimum(clusters[0], box[0])
    y = np.minimum(clusters[1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[0] * clusters[1]

    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    # iou_ = intersection / (box_area + cluster_area - intersection + 1e-10)

    return iou_


def avg_iou(boxes, clusters):
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def get_closest_dist(point, centroids):
    min_dist = math.inf  # 初始设为无穷大
    # print(centroids)
    for i, centroid in enumerate(centroids):
        # print(centroids)
        dist = 1 - iou_kpp(point, centroid)		# 点和当前每个中心点进行计算距离
        if dist < min_dist:
            min_dist = dist		# 注意我K-means++博客中的这句“指该点离中心点这一数组中所有中心点距离中的最短距离”
    return min_dist


def kpp_centers(data_set: list, k: int) -> list:
    """
    从数据集中返回 k 个对象可作为质心
    """
    cluster_centers = []
    cluster_centers.append(random.choice(data_set))
    d = [0 for _ in range(len(data_set))]
    #print(d)
    for _ in range(1, k):
        total = 0.0
        for i, point in enumerate(data_set):
            d[i] = get_closest_dist(point, cluster_centers) # 与最近一个聚类中心的距离
            total += d[i]
        total *= random.random()
        for i, di in enumerate(d): # 轮盘法选出下一个聚类中心；
            total -= di
            if total > 0:
                continue
            cluster_centers.append(data_set[i])
            break
    return cluster_centers



def kmeans(boxes, k, dist=np.median):
    rows = boxes.shape[0]

    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))

    np.random.seed()

    # the Forgy method will fail if the whole array contains the same rows
    clusters = kpp_centers(boxes, k)
    clusters = np.array(clusters)
    #clusters = boxes[np.random.choice(rows, k, replace=False)] 这是K-means的，两个切换注释下就行了

    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)  # iou很大则距离很小
        # 对每个标注框选择与其距离最接近的集群中心的标号作为所属类别的编号。
        nearest_clusters = np.argmin(distances, axis=1)     # axis=1表示沿着列的方向水平延伸

        if (last_clusters == nearest_clusters).all():
            break

        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)    # 给每类算均值新中心点

        last_clusters = nearest_clusters
        # print(last_clusters)

    return clusters


def parse_anno(annotation_path, target_size=None):
    anno = open(annotation_path, 'r',encoding='utf-8')
    result = []
    # 对每一个标记图片
    for line in anno:
        s = line.strip().split(' ')
        img_w = int(s[1])
        img_h = int(s[2])
        s = s[3:]
        box_cnt = len(s) // 5
        # 分别处理每一个标记框的信息，并提取标记框的高度和宽度，存入result 列表
        for i in range(box_cnt):
            x_min, y_min, x_max, y_max = float(s[i*5+1]), float(s[i*5+2]), float(s[i*5+3]), float(s[i*5+4])
            width = x_max - x_min
            height = y_max - y_min
            # assert width > 0
            # assert height > 0
            # use letterbox resize, i.e. keep the original aspect ratio
            # get k-means anchors on the resized target image size
            if target_size is not None:
                resize_ratio = min(target_size[0] / img_w, target_size[1] / img_h)
                width *= resize_ratio
                height *= resize_ratio
                result.append([width, height])
            # get k-means anchors on the original image size
            else:
                result.append([width, height])
    result = np.asarray(result)
    return result


def get_kmeans(anno, cluster_num=9):
    # 使用kmeans算法计算需要的anchors
    anchors = kmeans(anno, cluster_num)

    ave_iou = avg_iou(anno, anchors)
    # 格式化为int类型
    anchors = anchors.astype('int').tolist()
    # 按照面积大小排序，
    anchors = sorted(anchors, key=lambda x: x[0] * x[1])

    return anchors, ave_iou


if __name__ == '__main__':

    # ave_iou=0.1
    n=12

    target_size = [640, 640]
    annotation_path = "diorship-kmean++_test.txt"
    anno_result = parse_anno(annotation_path, target_size=target_size)
    anchors, ave_iou = get_kmeans(anno_result, n)

    # 格式化输出anchors数据
    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]

    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)
# while n<14:
#     target_size = [640, 640]
#     annotation_path = "gongjingkmean++_train.txt"
#     anno_result = parse_anno(annotation_path, target_size=target_size)
#     anchors, ave_iou = get_kmeans(anno_result, n)
#
#     # 格式化输出anchors数据
#     anchor_string = ''
#     for anchor in anchors:
#         anchor_string += '{},{}, '.format(anchor[0], anchor[1])
#     anchor_string = anchor_string[:-2]
#     print('n is')
#     print(n)
#     print('anchors are:')
#     print(anchor_string)
#     print('the average iou is:')
#     print(ave_iou)
#     # if n==11 :
#     #     n=n+2
#     # else:
#     n=n+1


