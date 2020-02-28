import glob

try:
    from colors import *

    print(yellow + 'Import Colors successful!' + end)
except ImportError:
    print('Unable to import Colors from source/colors.py')

try:
    from ic_now_source import *

    print(yellow + 'Source import successful can begin operation for Excecution Path' + end)
except ImportError:
    print('Unable to import ic_now_source import')

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
    detector.setModelPath(os.path.join(execution_path, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    print(yellow + 'Execution Path Completed, can begin to detect image information' + end)
except OSError:
    print(
        bold + red + 'Unable to complete the operation, please check line 26 -> os.path.join for more information' + '\n' +
        'Thank you.' + end)

# Get image folder location
path = 'images'
# listing files/used to help the user see what images are on the images folder
# link for more info: https://mkyong.com/python/python-how-to-list-all-files-in-a-directory/
for images_folder in os.walk(path):
    for image_folder in images_folder:
        if '.jpg' in image_folder:
            image_folder.append(os.path.join(images_folder, image_folder))
# Print image info
print(image_folder)

# Aquire User input on what image they want to scan for data
print('Key Legend: Image name.jpg or png -> Enter')

image_see = input('Please type in the full name & image format you would like to scan: ')

# User input turned into string
image_aquire = str(image_see)
# img completed statement
img_full_info = '_img_complete'
img_format = '.jpg'
# combine image info
img_complete = image_aquire + img_full_info + img_format

try:
    detections = detector.detectObjectsFromImage(
        # find the img
        input_image=os.path.join(execution_path, "images/" + image_aquire),
        # out put the image / rename it with a new name
        output_image_path=os.path.join(execution_path, "image_detection_complete/" + img_complete),
        minimum_percentage_probability=50,
    )


except OSError:
    print(bold + red + 'Cannot acquire os.path, please check lines 37-39 for any issues' + '\n' +
          'Thank you.' + end)

# just loops & tell the user the img name & the % of that img being that obj
# e.g -> tree 50% , or human 85%, city stop lights 77%, etc
try:
    for eachObject in detections:
        print(bold + yellow + 'Acquired Image Info Success! Following image info is the following: ' + '\n' + end)
        print(bold, blue, eachObject["name"], ": ", end, bold, purple, round(eachObject["percentage_probability"]), '%',
              end)



except Exception:
    print(bold + red + 'Unable to complete the process, please check lines 49-52 for any issues' + '\n' +
          'Thank you.' + end)
