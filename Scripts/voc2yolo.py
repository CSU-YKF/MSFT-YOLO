import xml.etree.ElementTree as ET
import os
from os import getcwd
from tqdm import tqdm

classes = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    in_file = 'NEU-DET/ANNOTATIONS/%s.xml' % (image_id)
    out_file = open('NEU-DET/labels/%s.txt' % (image_id), 'w')
    try:
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            bb = convert((w, h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    except ET.ParseError:
        print(f"Error parsing {in_file}")
    except Exception as e:
        print(f"Error processing {in_file}: {e}")
    finally:
        out_file.close()


if __name__ == "__main__":
    wd = getcwd()
    print(wd)
    if not os.path.exists('../datasets/NEU-DET/labels/'):
        os.makedirs('../datasets/NEU-DET/labels/')
    image_ids = [f.split('.')[0] for f in os.listdir('../datasets/NEU-DET/images') if f.endswith('.jpg')]
    for image_id in tqdm(image_ids):
        convert_annotation(image_id)
