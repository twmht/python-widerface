from wider.wider import WIDER
import matplotlib.pyplot as plt
import cv2

wider = WIDER('/home/tumh/python-wider/data/v1',
              '/home/tumh/python-wider/data/WIDER_train/images',
              'wider_face_train.mat')


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
