try:
    from colors import *
    print(yellow + 'Import Colors successful!' + end)
except:
    print('Unable to import Colors from source/colors.py')

try:
    from ic_now_source import *
    print(yellow + 'Source import successful can begin operation for Excecution Path' + end)
except:
    print('Unable to aquire ic_now_source import')



# Just the os getting the dir that is current at the moment
execution_path = os.getcwd()
detector = ObjectDetection()
# more info of RetinaNet -> https://medium.com/@14prakash/the-intuition-behind-retinanet-eb636755607d
# https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4
# You learn something every day, as long you are constantly going forward and using the info
# the objective will be accomplished, just takes time & dedication ^_*
detector.setModelTypeAsRetinaNet()
# get the path of .h5 / find location folder that holds resnet50_coco_best_v2.0.1.h5

try:
    detector.setModelPath( os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    print(yellow + 'Execution Path Completed, can begin to detect image information' + end)
except OSError:
    print(bold + red + 'Unable to complete the operation, please check line 26 -> os.path.join for more information' + '\n' +
          'Thank you.' + end)

try:
    detections = detector.detectObjectsFromImage(
        # find the img

        input_image=os.path.join(execution_path, "images/cat00.jpg"),
        # out put the image / rename it with a new name
        output_image_path=os.path.join(execution_path, "image_detection_complete/cat_image_detect.jpg"),
        minimum_percentage_probability=50,
    )
except OSError:
    print(bold + red + 'Cannot acquire os.path, please check lines 37-39 for any issues' + '\n' +
          'Thank you.' + end )

# just loops & tell the user the img name & the % of that img being that obj
# e.g -> tree 50% , or human 85%, city stop lights 77%, etc
try:
    for eachObject in detections:
        print(bold + yellow + 'Acquired Image Info Success! Following image info is the following: ' + '\n' + end)
        print(bold , blue , eachObject["name"], ": ", end, bold, purple, round(eachObject["percentage_probability"]),'%', end)


except Exception:
    print(bold + red + 'Unable to complete the process, please check lines 49-52 for any issues'+ '\n' +
          'Thank you.' + end )
