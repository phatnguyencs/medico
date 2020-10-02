from pycocotools.coco import COCO 
from .utils import save_coco, filter_annotations, save_images, create_dir
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import numpy as np 
import skimage.io as io 
import matplotlib.pyplot as plt 
import os
import funcy
import json
from sklearn.model_selection import train_test_split

class MyCOCO(COCO):
    def __init__(self, annot_file=None):
        COCO.__init__(self, annot_file)

    def showAnns(self, anns, category_map, draw_bbox=False, draw_mask=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                label = category_map[ann['category_id']]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon                        
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann['image_id']]
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], t['height'], t['width'])
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))
                if 'keypoints' in ann and type(ann['keypoints']) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann['category_id'])[0]['skeleton'])-1
                    kp = np.array(ann['keypoints'])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk]>0):
                            plt.plot(x[sk],y[sk], linewidth=3, color=c)
                    plt.plot(x[v>0], y[v>0],'o',markersize=8, markerfacecolor=c, markeredgecolor='k',markeredgewidth=2)
                    plt.plot(x[v>1], y[v>1],'o',markersize=8, markerfacecolor=c, markeredgecolor=c, markeredgewidth=2)

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
                    #print(f"bbox: {bbox_x}, {bbox_y}, {bbox_w}, {bbox_h}")
                    poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
                    np_poly = np.array(poly).reshape((4,2))
                    polygons.append(Polygon(np_poly, label=label))
                    color.append(c)
                    ax.text(bbox_x+bbox_w/2, bbox_y+bbox_h/2, label, ha='center', va='center', size=6)

            if draw_mask:
                p = PatchCollection(polygons, facecolor=color, linewidths=1, alpha=0.4)
                ax.add_collection(p)

            if draw_bbox:
                p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=1)
                ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])


class CocoInstance(object):
    def __init__(self, annot_dir):
        '''
        annot_dir: json file
        '''
        self.annot_dir = annot_dir 
        self.coco_instance = MyCOCO(annot_dir)

        ## Load cats
        self.cats = self.coco_instance.loadCats(self.coco_instance.getCatIds())
        self.nms = [cat['name'] for cat in self.cats]

        ## Load supercats
        self.super_cats = set([cat['supercategory'] for cat in self.cats])
        
        self.category_map = self._create_category_map()
        #self._modify_category_id()

    def _create_category_map(self):
        cat_map = dict()
        for cat_ins in self.coco_instance.dataset['categories']:
            cat_map[cat_ins['id']] = cat_ins['name']
        
        return cat_map

    def _modify_category_id(self):
        for i, annot_ins in enumerate(self.coco_instance.dataset['annotations']):
            annot_ins["category_id"] = category_map[annot_ins["category_id"]]
            self.coco_instance.dataset['annotations'][i] = annot_ins
        
        for i, cat_ins in enumerate(self.coco_instance.dataset['categories']):
            cat_ins['id'] = category_map[cat_ins['id']]
            self.coco_instance.dataset['categories'][i] = cat_ins
        
        for i, img_ins in enumerate(self.coco_instance.dataset['images']):
            for j in range(len(img_ins['category_ids'])):
                img_ins['category_ids'][j] = category_map[img_ins['category_ids'][j]]
            self.coco_instance.dataset['images'][i] = img_ins

    def getAnnotGivenImageId(self, img_id, cat_ids=None):
        if cat_ids is not None:
            annIds = self.coco_instance.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
        else:
            annIds = self.coco_instance.getAnnIds(imgIds=[img_id], iscrowd=None)
        return annIds

    def getAllCatsGivenImageId(self, img_id):
        return self.coco_instance.loadImgs([img_id])[0]['category_ids']

    def getCatNameGivenCatId(self, cat_ids: []):
        cat_nms = []
        for i in cat_ids:
            for dict_cat in self.coco_instance.dataset['categories']:
                if dict_cat['id'] == i:
                    cat_nms.append(dict_cat['name'])

        return cat_nms

    def getImageInstanceGivenFilename(self, filename):
        for img_ins in self.coco_instance.dataset['images']:
            if '/' in filename:
                if img_ins['path'] == filename:
                    return img_ins
            else:
                if img_ins['file_name'] == filename:
                    return img_ins

        return None

    def showImageWithMask(self, img_dict, data_dir, annot_id=None):
        fig, axes = plt.subplots(figsize=(8, 8), nrows=1, ncols=2, squeeze=False)

        # set figure name
        fig.canvas.set_window_title(f"[{','.join(self.getCatNameGivenCatId(img_dict['category_ids']))}]: {img_dict['file_name']}")
        fig.subplots_adjust(wspace=0.3)
        fig.subplots_adjust(hspace=0.3)

        I = io.imread(os.path.join(data_dir, img_dict['file_name']))
        axes[0,0].imshow(I)
        axes[0,0].axis('off')

        axes[0,1].imshow(I)
        axes[0,1].axis('off')
        

        if annot_id is None:
            annIds = self.getAnnotGivenImageId(img_id = img_dict['id'], cat_ids = img_dict['category_ids'])

        anns = self.coco_instance.loadAnns(annot_id)
        self.coco_instance.showAnns(anns, category_map=self.category_map, draw_bbox=True, draw_mask=False)
    
        plt.show()

    def comparePredictionWithLabel(self, img_filename, pred_datadir, raw_datadir, save_dir = None):
        fig, axes = plt.subplots(figsize=(10, 10), nrows=1, ncols=3, squeeze=False)
        fig.subplots_adjust(wspace=0.3)
        fig.subplots_adjust(hspace=0.3)

        I = io.imread(os.path.join(raw_datadir, img_filename))

        # show raw image
        axes[0,0].imshow(I)
        axes[0,0].axis('off')

        #show prediction result
        try:
            predI = io.imread(os.path.join(pred_datadir, img_filename))
        except Exception:
            print(f"cannot open image {img_filename}")
            return

        axes[0,1].imshow(predI)
        axes[0,1].axis('off')

        # show image with labels
        axes[0,2].imshow(I)
        axes[0,2].axis('off')

        img_id = None
        img_ins = None
        for img in self.coco_instance.dataset['images']:
            if img['file_name'] == img_filename:
                img_id = img['id']
                img_ins = img
                break
        
        if img_id is None:
            print(f"cannot find img_id for img_name: {img_filename}")
            return
        
        annot_id = self.getAnnotGivenImageId(img_id = img_id)
        anns = self.coco_instance.loadAnns(annot_id)
        # if len(anns) == 1:
        #     print(f"{img_filename}")

        cat_names = self.getCatNameGivenCatId(self.getAllCatsGivenImageId(img_id))

        self.coco_instance.showAnns(anns, category_map=self.category_map, draw_bbox=True, draw_mask=True)
        
        if save_dir is not None:
            create_dir(save_dir)
            # plt.savefig(os.path.join(save_dir, img_filename))

            for c in cat_names:
                cat_save_dir = os.path.join(save_dir, c)
                create_dir(cat_save_dir)
                plt.savefig(os.path.join(cat_save_dir, img_filename))
            
            plt.clf()

    def getImagesGivenCats(self, list_cats=[], visualize=True, data_dir=None):
        catIds = self.coco_instance.getCatIds(catNms = list_cats)
        imgIds = self.coco_instance.getImgIds(catIds = catIds)
        #imgIds = self.coco_instance.getImgIds(imgIds = [0])
        
        random_idx = np.random.randint(0, len(imgIds))
        # print(f'random index: {random_idx}')
        # print(f'random imgid: {imgIds[random_idx]}, {type(imgIds[random_idx])}')

        img = self.coco_instance.loadImgs([imgIds[random_idx]])[0]


        if visualize and data_dir is not None:
            fig, axes = plt.subplots(figsize=(10, 8), nrows=1, ncols=2, squeeze=False)

            fig.canvas.set_window_title(f"[{','.join(self.getCatNameGivenCatId(img['category_ids']))}]: {img['file_name']}")

            fig.subplots_adjust(wspace=0.3)
            fig.subplots_adjust(hspace=0.3)

            I = io.imread(os.path.join(data_dir, img['file_name']))
            axes[0,0].imshow(I)
            axes[0,0].axis('off')

            axes[0,1].imshow(I)
            axes[0,1].axis('off')

            annIds = self.getAnnotGivenImageId(img_id = imgIds[random_idx], 
                                            cat_ids = self.getAllCatsGivenImageId(imgIds[random_idx])
                                            )
            anns = self.coco_instance.loadAnns(annIds)
            self.coco_instance.showAnns(anns, self.category_map, draw_mask=True, draw_bbox=True)
            
            plt.show()
            
        return img

    def showImageGivenFilename(self, img_filename: str, data_dir: str, need_map=True):
        img_id = None
        img_dict = None
        for img in self.coco_instance.dataset['images']:
            if img['file_name'] == img_filename:
                img_id = img['id']
                img_dict = img
                break
        
        if need_map and img_id is not None:
            annot_id = self.getAnnotGivenImageId(img_id, )
            self.showImageWithMask(img_dict, data_dir, annot_id)
            
    def getCats(self):
        print(f"Categories: {', '.join(self.nms)}")
        return self.nms

    def getSupercats(self):
        print(f"SuperCategories: {', '.join(self.super_cats)}")
        return self.super_cats

    def get_info_for_given_images(self, list_images):
        list_annots = []
        list_ins = []
        for img in list_images:
            img_ins = self.getImageInstanceGivenFilename(img)
            if img_ins is not None:
                list_ins.append(img_ins)
                img_id = img_ins['id']

                for annot_ins in self.coco_instance.dataset['annotations']:
                    if annot_ins['image_id'] == img_id:
                        list_annots.append(annot_ins)
                        # break

        return list_ins, list_annots

    def split_coco(self, test_size, save_dir, image_dir, having_annots=True, save_image=False):
        '''
        having_annots: True if we use only images have labels. False otherwise
        '''
        info = None#self.coco_instance.dataset['info']
        licenses = None#self.coco_instance.dataset['licenses']
        
        images, annots = self.get_info_for_given_images(os.listdir(image_dir))
        cats = self.coco_instance.dataset['categories']

        n_images = len(images)
        print(f"all_images: {n_images}")
        
        imgs_with_annots = funcy.lmap(lambda pair: int(pair['image_id']), annots)

        if having_annots:
            images = funcy.lremove(lambda pair: int(pair['id']) not in imgs_with_annots, images)
        
        print(f"n_samples: {len(images)}")
        train, val = train_test_split(images, test_size=test_size)
        train_dir = os.path.join(save_dir, 'train_annot.json')
        val_dir = os.path.join(save_dir, 'val_annot.json')

        save_coco(train_dir, info, licenses, train, filter_annotations(annots, train), cats)
        save_coco(val_dir, info, licenses, val, filter_annotations(annots, val), cats)

        if save_image:
            save_images(train, data_dir = image_dir, save_dir = os.path.join(save_dir, 'train') )
            save_images(val, data_dir = image_dir, save_dir = os.path.join(save_dir, 'val') )



    # Create annot json file given list images
    def create_annot_file_given_image_name(self, image_dir: str, save_path: str):
        list_ins = []
        list_annots = []
        fail_img_name= []
        for image_name in os.listdir(image_dir):
            img_ins = self.getImageInstanceGivenFilename(image_name)
            

            ann_ids = self.getAnnotGivenImageId(img_ins['id'])
            ann_ins = self.coco_instance.loadAnns(ann_ids)
            
            if len(ann_ins) == 0:
                continue

            list_ins.append(img_ins)
            list_annots.extend(ann_ins)
        
        cats = self.coco_instance.dataset['categories']
        info = None
        licenses = None
        save_coco(save_path, info, licenses, list_ins, list_annots, cats)
    
    # -----------------------------------------------------------------------------
    # split and save hard cases (including many classes)

    def split_hard_cases(self, image_dir: str, save_img_dir: str, save_annot_dir: str, folder_name = 'hard_cases'):
        '''
            Extract hard cases (have many classes in a single image) to save_dir
        '''
        list_ins = []
        list_annots = []
        list_images = os.listdir(image_dir)
        # list_ins, list_annots = self.get_info_for_given_images(os.listdir(image_dir))

        for i, img_ins in enumerate(self.coco_instance.dataset['images']):
            if img_ins['file_name'] not in list_images:
                continue

            n_cat = len(img_ins['category_ids'])
            if n_cat >= 2:
                ann_ids = self.getAnnotGivenImageId(img_ins['id'])
                # if len(ann_ids) != n_cat:
                #     print(f"{img_ins['file_name']}, {len(ann_ids)}")

                ann_ins = self.coco_instance.loadAnns(ann_ids)
                if len(ann_ins) == 0:
                    continue

                list_ins.append(img_ins)
                list_annots.extend(ann_ins)
        
        print(f"n images: {len(list_ins)}")
        print(f"n annots: {len(list_annots)}")


        cats = self.coco_instance.dataset['categories']
        info = None
        licenses = None

        save_annot_path = os.path.join(save_annot_dir, f"{folder_name}.json")
        save_coco(save_annot_path, info, licenses, list_ins, list_annots, cats)
        save_images(list_ins, image_dir, os.path.join(save_img_dir, folder_name))

