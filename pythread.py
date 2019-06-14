import threading
import queue
# Multi threads processing
im_quene = queue.Queue(maxsize=0)


ids = ['we','are','good']
for fimage in dataset:
    im_quene.put(fimage)

def process():
    # while not im_quene.empty():
    #     im_i = im_quene.get()
    #     print(im_i)
    while not im_quene.empty():
        fimage = im_quene.get()
        if nrof_images_total % 100 == 0:
            print("Processing %d, (%s)" % (nrof_images_total, nrof))
        nrof_images_total += 1
        # if nrof_images_total<950000:
        #  continue
        image_path = fimage.image_path
        if not os.path.exists(image_path):
            print('image not found (%s)' % image_path)
            continue
        filename = os.path.splitext(os.path.split(image_path)[1])[0]
        # print(image_path)
        try:
            img = misc.imread(image_path)
        except (IOError, ValueError, IndexError) as e:
            errorMessage = '{}: {}'.format(image_path, e)
            print(errorMessage)
        else:
            if img.ndim < 2:
                print('Unable to align "%s", img dim error' % image_path)
                # text_file.write('%s\n' % (output_filename))
                continue
            if img.ndim == 2:
                img = to_rgb(img)
            img = img[:, :, 0:3]
            _paths = fimage.image_path.split('/')
            a, b = _paths[-2], _paths[-1]
            target_dir = os.path.join(args.output_dir, a)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            target_file = os.path.join(target_dir, b)
            _minsize = minsize
            _bbox = None
            _landmark = None
            bounding_boxes, points = detect_face.detect_face(img, _minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(img.shape)[0:2]
                bindex = 0
                if nrof_faces > 1:
                    bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
                    img_center = img_size / 2
                    offsets = np.vstack(
                        [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
                    offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
                    bindex = np.argmax(
                        bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
                _bbox = bounding_boxes[bindex, 0:4]
                _landmark = points[:, bindex].reshape((2, 5)).T
                nrof[0] += 1
            else:
                nrof[1] += 1
            warped = face_preprocess.preprocess(img, bbox=_bbox, landmark=_landmark, image_size=args.image_size)
            bgr = warped[..., ::-1]
            # print(bgr.shape)
            cv2.imwrite(target_file, bgr)
            oline = '%d\t%s\t%d\n' % (1, target_file, int(fimage.classname))
            text_file.write(oline)

dataset = face_image.get_dataset('lfw', args.input_dir)
threads = [threading.Thread(target=process, args=()) for i in range(2)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()