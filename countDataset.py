import glob

trainFilenamesList = glob.glob('C://Users//fdquinones//Documents//Projects//Utpl-maestria//facemask//mask_yolo_test//*.txt')

totalMask = 0
totalNoMask = 0
totalMaskIncorrect = 0

print (trainFilenamesList)

 

for f in trainFilenamesList:
    print("File: [{}]".format(f))
    # Using readlines()
    file = open(f, 'r')
    lines = file.readlines()
    # Strips the newline character
    count = 0
    for line in lines:
        count += 1
        print("Line{}: {}".format(count, line.strip()))
        if line.startswith("0"):
            totalMask += 1
        if line.startswith("1"):
            totalNoMask += 1
        if line.startswith("2"):
            totalMaskIncorrect += 1

print("TOTAL CON MASCARILLA: [{}]".format(totalMask))
print("TOTAL SIN MASCARILLA: [{}]".format(totalNoMask))
print("TOTAL MASCARILLA INCORRECTA: [{}]".format(totalMaskIncorrect))