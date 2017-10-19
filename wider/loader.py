import matplotlib.pyplot as plt
import os
import scipy.io


class DATA(object):
    def __init__(self, image_name, bboxes):
        self.image_name = image_name
        self.bboxes = bboxes


class WIDER(object):
    """
    Build a wider parser

    Parameters
    ----------
    path_to_label : path of the label file
    path_to_image : path of the image files
    fname : name of the label file

    Returns
    -------
    a wider parser
    """
    def __init__(self, path_to_label, path_to_image, fname):
        self.path_to_label = path_to_label
        self.path_to_image = path_to_image

        self.f = scipy.io.loadmat(os.path.join(path_to_label, fname))
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')

    def next(self):
        for event_idx, event in enumerate(self.event_list):
            directory = event[0][0]
            for im_idx, im in enumerate(self.file_list[event_idx][0]):
                im_name = im[0][0]
                face_bbx = self.face_bbx_list[event_idx][0][im_idx][0]
                #  print face_bbx.shape

                bboxes = []

                for i in range(face_bbx.shape[0]):
                    xmin = int(face_bbx[i][0])
                    ymin = int(face_bbx[i][1])
                    xmax = int(face_bbx[i][2]) + xmin
                    ymax = int(face_bbx[i][3]) + ymin
                    bboxes.append((xmin, ymin, xmax, ymax))

                yield DATA(os.path.join(self.path_to_image, directory,
                           im_name + '.jpg'), bboxes)
