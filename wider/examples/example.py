from wider import WIDER
import matplotlib.pyplot as plt
import cv2

# arg1: path to label
# arg2: path to images
# arg3: label file name
wider = WIDER('/home/tumh/wider_face_split',
              '/home/tumh/WIDER_val/images',
              'wider_face_val.mat')


# press ctrl-C to stop the process
for data in wider.next():

    im = cv2.imread(data.image_name)

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for bbox in data.bboxes:

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()
