#---------------------------------------------#
#   运行前一定要修改classes
#   如果生成的2007_train.txt里面没有目标信息
#   那么就是因为classes没有设定正确
#---------------------------------------------#
import xml.etree.ElementTree as ET
from os import getcwd

sets=[('train'),  ('test')]
#-----------------------------------------------------#
#   这里设定的classes顺序要和model_data里的txt一样
#-----------------------------------------------------#
classes = ["ship"]

def convert_annotation(image_id, list_file):
    in_file = open(r'/user-data/diorship/xml/%s.xml'%(image_id), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()
    h = int(tree.findtext('./size/height'))
    w= int(tree.findtext('./size/width'))
    list_file.write(" " + str(w) + " " + str(h)  )
    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(' '+str(cls_id)+' '+" ".join([str(a) for a in b]))

wd = getcwd()

for image_set in sets:
    image_ids = open(r'/user-data/diorship/ceshi/%s.txt' % (image_set), encoding='utf-8').read().strip().split()
    list_file = open('diorship-kmean++_%s.txt' % (image_set), 'w', encoding='utf-8')
    for image_id in image_ids:
        list_file.write(r'/user-data/diorship/image/%s.jpg'%(image_id))
        convert_annotation(image_id, list_file)
        list_file.write('\n')
    list_file.close()
